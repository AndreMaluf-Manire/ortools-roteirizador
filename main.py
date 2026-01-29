"""
Microserviço de Otimização de Rotas com OR-Tools
Roteirizador Manirê / Fruleve

VERSÃO 4.0 - Correções:
- Velocidade média configurável (padrão 16 km/h para entregas urbanas)
- Propagação correta de tempo (departure → próxima chegada)
- Cálculo correto de arrived_early e arrived_late
- Distância real calculada e retornada
- Service_time de cada cliente respeitado
- Penalidade alta para não deixar entregas órfãs
"""

from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import os
import math
import time

app = FastAPI(
    title="OR-Tools Route Optimizer",
    description="API de otimização de rotas para o Roteirizador Manirê",
    version="4.0.0"
)

# CORS para permitir chamadas do Lovable/Supabase
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Chave de API para autenticação
API_KEY = os.getenv("ORTOOLS_API_KEY", "dev-key-change-in-production")


# ============== MODELOS DE DADOS ==============

class Location(BaseModel):
    lat: float
    lng: float


class Depot(BaseModel):
    id: str
    name: str
    location: Location


class Customer(BaseModel):
    id: str
    name: str
    location: Location
    window_start: Optional[int] = None  # Minutos desde meia-noite (ex: 480 = 08:00)
    window_end: Optional[int] = None    # Minutos desde meia-noite (ex: 720 = 12:00)
    service_time: int = 15              # Tempo de atendimento em minutos


class Delivery(BaseModel):
    id: str
    customer_id: str
    boxes: int = 0
    weight_kg: float = 0
    value: float = 0  # Valor em R$


class Vehicle(BaseModel):
    id: str
    name: str
    capacity_boxes: int
    capacity_kg: float = 99999


class OptimizeRequest(BaseModel):
    depot: Depot
    customers: List[Customer]
    deliveries: List[Delivery]
    vehicles: List[Vehicle]
    time_matrix: Optional[List[List[int]]] = None  # Matriz de tempos em minutos (opcional)
    distance_matrix: Optional[List[List[float]]] = None  # Matriz de distâncias em km (opcional)
    start_time: int = 360         # Horário de início em minutos (default 06:00)
    max_route_duration: int = 720 # Duração máxima da rota em minutos (default 12h)
    mode: str = "minimize_vehicles"
    average_speed_kmh: float = 16.0  # NOVO: Velocidade média em km/h (padrão 16 como RoutEasy)


class Stop(BaseModel):
    delivery_id: str
    customer_id: str
    customer_name: str
    stop_order: int
    arrival_time: int           # Quando o carro CHEGA (minutos desde meia-noite)
    effective_start: int        # Quando COMEÇA a entregar (pode ser diferente se chegou cedo)
    departure_time: int         # Quando o carro SAI
    service_time: int           # Tempo de atendimento usado
    wait_time: int = 0          # Tempo esperando cliente abrir
    travel_time_from_prev: int = 0  # Tempo de viagem do ponto anterior
    distance_from_prev_km: float = 0  # Distância do ponto anterior
    window_start: Optional[int]
    window_end: Optional[int]
    window_ok: bool
    arrived_early: bool = False  # Chegou ANTES da janela
    arrived_late: bool = False   # Chegou DEPOIS da janela
    boxes: int
    weight_kg: float
    value: float = 0  # Valor em R$


class Route(BaseModel):
    vehicle_id: Optional[str]
    vehicle_name: Optional[str]
    stops: List[Stop]
    total_boxes: int
    total_weight_kg: float
    total_value: float = 0  # NOVO: Valor total em R$
    total_time_minutes: int
    total_distance_km: float
    total_wait_time: int = 0
    total_service_time: int = 0
    total_travel_time: int = 0
    capacity_used_percent: float = 0


class OptimizeResponse(BaseModel):
    success: bool
    message: str
    routes: List[Route]
    unassigned_deliveries: List[str]
    vehicles_used: int
    total_deliveries: int
    total_value: float = 0  # NOVO: Valor total geral em R$
    optimization_time_ms: int


# ============== FUNÇÕES AUXILIARES ==============

def haversine_distance(loc1: Location, loc2: Location) -> float:
    """Calcula distância em km entre dois pontos usando fórmula de Haversine"""
    R = 6371  # Raio da Terra em km
    lat1, lon1 = math.radians(loc1.lat), math.radians(loc1.lng)
    lat2, lon2 = math.radians(loc2.lat), math.radians(loc2.lng)
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    return R * c


def get_location_for_node(node: int, request: 'OptimizeRequest', customer_map: dict) -> Location:
    """Retorna a localização de um nó (0 = depot, >0 = delivery)"""
    if node == 0:
        return request.depot.location
    else:
        delivery = request.deliveries[node - 1]
        customer = customer_map.get(delivery.customer_id)
        return customer.location if customer else request.depot.location


def get_travel_time(from_node: int, to_node: int, request: 'OptimizeRequest', customer_map: dict) -> int:
    """
    Retorna tempo de viagem entre dois nós em minutos.
    Usa matriz de tempos se disponível, senão calcula com velocidade média.
    """
    # Tentar usar matriz de tempos (Google Distance Matrix)
    if request.time_matrix:
        if len(request.time_matrix) > from_node and len(request.time_matrix[from_node]) > to_node:
            return request.time_matrix[from_node][to_node]
    
    # Fallback: calcular baseado em distância e velocidade média
    loc1 = get_location_for_node(from_node, request, customer_map)
    loc2 = get_location_for_node(to_node, request, customer_map)
    distance_km = haversine_distance(loc1, loc2)
    
    # Usar velocidade média configurada (padrão 16 km/h para entregas urbanas)
    speed_kmh = request.average_speed_kmh if request.average_speed_kmh > 0 else 16.0
    travel_time_minutes = (distance_km / speed_kmh) * 60
    
    return max(1, int(travel_time_minutes))  # Mínimo 1 minuto


def get_distance(from_node: int, to_node: int, request: 'OptimizeRequest', customer_map: dict) -> float:
    """Retorna distância em km entre dois nós"""
    # Tentar usar matriz de distâncias
    if request.distance_matrix:
        if len(request.distance_matrix) > from_node and len(request.distance_matrix[from_node]) > to_node:
            return request.distance_matrix[from_node][to_node]
    
    # Fallback: calcular com haversine
    loc1 = get_location_for_node(from_node, request, customer_map)
    loc2 = get_location_for_node(to_node, request, customer_map)
    return haversine_distance(loc1, loc2)


def minutes_to_time(minutes: int) -> str:
    """Converte minutos desde meia-noite para formato HH:MM"""
    hours = minutes // 60
    mins = minutes % 60
    return f"{hours:02d}:{mins:02d}"


# ============== SOLVER OR-TOOLS ==============

def solve_vrptw(request: OptimizeRequest) -> OptimizeResponse:
    """
    Resolve o problema CVRPTW usando OR-Tools.
    
    VERSÃO 4.0:
    - Velocidade média configurável (padrão 16 km/h)
    - Propagação correta de tempo
    - Cálculo correto de arrived_early/arrived_late
    """
    start_time_ms = time.time() * 1000
    
    # Mapas para acesso rápido
    customer_map = {c.id: c for c in request.customers}
    delivery_map = {d.id: d for d in request.deliveries}
    
    # Nós: depot (0) + entregas (1 a N)
    nodes = ["depot"] + [d.id for d in request.deliveries]
    num_locations = len(nodes)
    
    if num_locations <= 1:
        return OptimizeResponse(
            success=True,
            message="Nenhuma entrega para roteirizar",
            routes=[],
            unassigned_deliveries=[],
            vehicles_used=0,
            total_deliveries=0,
            total_value=0,
            optimization_time_ms=int(time.time() * 1000 - start_time_ms)
        )
    
    # Configurar veículos
    if request.mode == "minimize_vehicles":
        num_vehicles = len(request.deliveries)
        vehicles = []
        for i in range(num_vehicles):
            if i < len(request.vehicles):
                vehicles.append(request.vehicles[i])
            else:
                avg_capacity = sum(v.capacity_boxes for v in request.vehicles) // len(request.vehicles) if request.vehicles else 100
                vehicles.append(Vehicle(
                    id=f"extra_{i}",
                    name=f"Veículo Extra {i - len(request.vehicles) + 1}",
                    capacity_boxes=avg_capacity,
                    capacity_kg=99999
                ))
    else:
        num_vehicles = len(request.vehicles)
        vehicles = request.vehicles
    
    if num_vehicles == 0:
        return OptimizeResponse(
            success=False,
            message="Nenhum veículo disponível",
            routes=[],
            unassigned_deliveries=[d.id for d in request.deliveries],
            vehicles_used=0,
            total_deliveries=len(request.deliveries),
            total_value=0,
            optimization_time_ms=int(time.time() * 1000 - start_time_ms)
        )
    
    # Criar modelo OR-Tools
    manager = pywrapcp.RoutingIndexManager(num_locations, num_vehicles, 0)
    routing = pywrapcp.RoutingModel(manager)
    
    # ===== CALLBACK DE TEMPO (inclui service_time) =====
    def time_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        
        # Tempo de viagem
        travel_time = get_travel_time(from_node, to_node, request, customer_map)
        
        # Adicionar tempo de serviço do nó de origem (exceto depot)
        service_time = 0
        if from_node > 0:
            delivery = request.deliveries[from_node - 1]
            customer = customer_map.get(delivery.customer_id)
            service_time = customer.service_time if customer else 15
        
        return travel_time + service_time
    
    transit_callback_index = routing.RegisterTransitCallback(time_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    
    # ===== DIMENSÃO DE TEMPO =====
    routing.AddDimension(
        transit_callback_index,
        300,  # Slack máximo 5h (para espera em janelas)
        request.max_route_duration + 300,
        False,
        "Time"
    )
    time_dimension = routing.GetDimensionOrDie("Time")
    
    # Configurar janelas de tempo
    for location_idx in range(num_locations):
        index = manager.NodeToIndex(location_idx)
        
        if location_idx == 0:
            # Depot: pode sair a qualquer momento
            time_dimension.CumulVar(index).SetRange(0, request.max_route_duration)
        else:
            delivery = request.deliveries[location_idx - 1]
            customer = customer_map.get(delivery.customer_id)
            
            if customer and customer.window_start is not None and customer.window_end is not None:
                # Janela relativa ao início da rota
                window_start_rel = max(0, customer.window_start - request.start_time)
                window_end_rel = max(window_start_rel + 60, customer.window_end - request.start_time + 120)
                time_dimension.CumulVar(index).SetRange(window_start_rel, window_end_rel)
            else:
                time_dimension.CumulVar(index).SetRange(0, request.max_route_duration)
    
    # ===== DIMENSÃO DE CAPACIDADE =====
    def demand_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        if from_node == 0:
            return 0
        delivery = request.deliveries[from_node - 1]
        return delivery.boxes
    
    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,
        [v.capacity_boxes for v in vehicles],
        True,
        "Capacity"
    )
    
    # ===== PENALIDADES =====
    if request.mode == "minimize_vehicles":
        for vehicle_idx in range(num_vehicles):
            routing.SetFixedCostOfVehicle(10000, vehicle_idx)
    
    # Penalidade MUITO alta por não entregar (forçar todas as entregas)
    penalty = 1000000
    for node in range(1, num_locations):
        routing.AddDisjunction([manager.NodeToIndex(node)], penalty)
    
    # ===== PARÂMETROS DE BUSCA =====
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_parameters.time_limit.FromSeconds(30)
    
    # ===== RESOLVER =====
    solution = routing.SolveWithParameters(search_parameters)
    
    if not solution:
        return OptimizeResponse(
            success=False,
            message="Não foi possível encontrar uma solução. Verifique janelas e capacidades.",
            routes=[],
            unassigned_deliveries=[d.id for d in request.deliveries],
            vehicles_used=0,
            total_deliveries=len(request.deliveries),
            total_value=0,
            optimization_time_ms=int(time.time() * 1000 - start_time_ms)
        )
    
    # ===== EXTRAIR SOLUÇÃO (VERSÃO 4.0 - PROPAGAÇÃO CORRETA) =====
    routes = []
    vehicles_used = 0
    total_value_all = 0
    
    for vehicle_idx in range(num_vehicles):
        index = routing.Start(vehicle_idx)
        
        # Coletar sequência de nós
        route_nodes = []
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            if node > 0:
                route_nodes.append(node)
            index = solution.Value(routing.NextVar(index))
        
        if not route_nodes:
            continue
        
        # Calcular tempos de forma PROPAGADA
        stops = []
        route_boxes = 0
        route_weight = 0.0
        route_value = 0.0
        route_distance = 0.0
        route_wait_time = 0
        route_service_time = 0
        route_travel_time = 0
        
        # Começar do depot no horário de início
        current_time = request.start_time
        prev_node = 0  # depot
        
        for stop_order, node in enumerate(route_nodes, 1):
            delivery = request.deliveries[node - 1]
            customer = customer_map.get(delivery.customer_id)
            
            # Calcular tempo de viagem e distância do ponto anterior
            travel_time = get_travel_time(prev_node, node, request, customer_map)
            distance = get_distance(prev_node, node, request, customer_map)
            
            # Horário de chegada = horário atual + tempo de viagem
            arrival_time = current_time + travel_time
            
            # Tempo de serviço deste cliente
            service_time = customer.service_time if customer else 15
            
            # ===== VERIFICAR JANELA E CALCULAR ESPERA =====
            wait_time = 0
            arrived_early = False
            arrived_late = False
            effective_start = arrival_time
            
            if customer and customer.window_start is not None:
                if arrival_time < customer.window_start:
                    # Chegou ANTES de abrir - ESPERA até abrir
                    wait_time = customer.window_start - arrival_time
                    effective_start = customer.window_start
                    arrived_early = True
                    print(f"[ADIANTADO] {customer.name}: Chega {minutes_to_time(arrival_time)}, Abre {minutes_to_time(customer.window_start)}, Espera {wait_time}min")
                
                if customer.window_end is not None and arrival_time > customer.window_end:
                    # Chegou DEPOIS de fechar - ATRASADO
                    arrived_late = True
                    print(f"[ATRASADO] {customer.name}: Chega {minutes_to_time(arrival_time)}, Fecha {minutes_to_time(customer.window_end)}")
            
            # Horário de partida = início efetivo + tempo de serviço
            departure_time = effective_start + service_time
            
            # Verificar se está dentro da janela (para window_ok)
            window_ok = True
            if customer and customer.window_start is not None and customer.window_end is not None:
                # Considera OK se chegou dentro da janela OU chegou cedo (vai esperar)
                window_ok = not arrived_late
            
            stops.append(Stop(
                delivery_id=delivery.id,
                customer_id=delivery.customer_id,
                customer_name=customer.name if customer else "Desconhecido",
                stop_order=stop_order,
                arrival_time=arrival_time,
                effective_start=effective_start,
                departure_time=departure_time,
                service_time=service_time,
                wait_time=wait_time,
                travel_time_from_prev=travel_time,
                distance_from_prev_km=round(distance, 2),
                window_start=customer.window_start if customer else None,
                window_end=customer.window_end if customer else None,
                window_ok=window_ok,
                arrived_early=arrived_early,
                arrived_late=arrived_late,
                boxes=delivery.boxes,
                weight_kg=delivery.weight_kg,
                value=delivery.value
            ))
            
            # Acumular totais
            route_boxes += delivery.boxes
            route_weight += delivery.weight_kg
            route_value += delivery.value
            route_distance += distance
            route_wait_time += wait_time
            route_service_time += service_time
            route_travel_time += travel_time
            
            # ===== ATUALIZAR HORÁRIO ATUAL PARA O PRÓXIMO PONTO =====
            # O próximo ponto começa a partir do DEPARTURE deste
            current_time = departure_time
            prev_node = node
        
        # Adicionar distância de volta ao depot
        distance_to_depot = get_distance(prev_node, 0, request, customer_map)
        route_distance += distance_to_depot
        
        # Tempo total da rota (do início até última saída)
        total_time = stops[-1].departure_time - request.start_time if stops else 0
        
        # Calcular ocupação
        vehicle = vehicles[vehicle_idx] if vehicle_idx < len(vehicles) else None
        capacity_used_percent = 0
        if vehicle and vehicle.capacity_boxes > 0:
            capacity_used_percent = round((route_boxes / vehicle.capacity_boxes) * 100, 1)
        
        vehicles_used += 1
        total_value_all += route_value
        
        routes.append(Route(
            vehicle_id=vehicle.id if vehicle and not vehicle.id.startswith("extra_") else None,
            vehicle_name=vehicle.name if vehicle else f"Rota Extra {vehicle_idx + 1}",
            stops=stops,
            total_boxes=route_boxes,
            total_weight_kg=round(route_weight, 2),
            total_value=round(route_value, 2),
            total_time_minutes=total_time,
            total_distance_km=round(route_distance, 2),
            total_wait_time=route_wait_time,
            total_service_time=route_service_time,
            total_travel_time=route_travel_time,
            capacity_used_percent=capacity_used_percent
        ))
    
    # Identificar entregas não alocadas
    assigned_deliveries = set()
    for route in routes:
        for stop in route.stops:
            assigned_deliveries.add(stop.delivery_id)
    
    unassigned = [d.id for d in request.deliveries if d.id not in assigned_deliveries]
    
    return OptimizeResponse(
        success=True,
        message=f"Otimização concluída: {vehicles_used} veículos, {len(assigned_deliveries)} entregas alocadas",
        routes=routes,
        unassigned_deliveries=unassigned,
        vehicles_used=vehicles_used,
        total_deliveries=len(request.deliveries),
        total_value=round(total_value_all, 2),
        optimization_time_ms=int(time.time() * 1000 - start_time_ms)
    )


# ============== ENDPOINTS ==============

@app.get("/")
async def root():
    return {"status": "ok", "service": "OR-Tools Route Optimizer", "version": "4.0.0"}


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "ortools_version": "9.x",
        "api_key_configured": API_KEY != "dev-key-change-in-production",
        "version": "4.0.0",
        "features": [
            "configurable average_speed_kmh (default 16)",
            "propagated time calculation",
            "service_time per customer",
            "wait_time when arrived early",
            "real distance calculation",
            "arrived_early/arrived_late flags",
            "capacity_used_percent",
            "total_value in R$",
            "increased penalty for unassigned"
        ]
    }


@app.post("/optimize", response_model=OptimizeResponse)
async def optimize_routes(
    request: OptimizeRequest,
    authorization: Optional[str] = Header(None)
):
    """
    Otimiza rotas usando OR-Tools CVRPTW.
    
    VERSÃO 4.0:
    - average_speed_kmh: Velocidade média configurável (padrão 16 km/h)
    - Tempo PROPAGADO: próxima entrega usa departure da anterior
    - service_time de cada cliente
    - wait_time quando chega antes de abrir
    - Distância real calculada
    - Valor em R$ por entrega e total
    """
    
    # Validar API Key
    if API_KEY != "dev-key-change-in-production":
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="API Key não fornecida")
        
        provided_key = authorization.replace("Bearer ", "")
        if provided_key != API_KEY:
            raise HTTPException(status_code=401, detail="API Key inválida")
    
    if not request.deliveries:
        return OptimizeResponse(
            success=True,
            message="Nenhuma entrega para roteirizar",
            routes=[],
            unassigned_deliveries=[],
            vehicles_used=0,
            total_deliveries=0,
            total_value=0,
            optimization_time_ms=0
        )
    
    if not request.vehicles and request.mode == "fixed_fleet":
        raise HTTPException(
            status_code=400,
            detail="Modo 'fixed_fleet' requer pelo menos um veículo"
        )
    
    try:
        result = solve_vrptw(request)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na otimização: {str(e)}")


@app.post("/validate")
async def validate_request(request: OptimizeRequest):
    """Valida os dados antes de otimizar."""
    issues = []
    
    customer_ids = {c.id for c in request.customers}
    for d in request.deliveries:
        if d.customer_id not in customer_ids:
            issues.append(f"Delivery {d.id} referencia customer inexistente: {d.customer_id}")
    
    for c in request.customers:
        if not (-90 <= c.location.lat <= 90) or not (-180 <= c.location.lng <= 180):
            issues.append(f"Customer {c.id} tem coordenadas inválidas")
    
    total_capacity = sum(v.capacity_boxes for v in request.vehicles) if request.vehicles else 0
    total_demand = sum(d.boxes for d in request.deliveries)
    total_value = sum(d.value for d in request.deliveries)
    
    if total_demand > total_capacity and request.mode == "fixed_fleet":
        issues.append(f"Demanda total ({total_demand} caixas) excede capacidade total ({total_capacity} caixas)")
    
    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "summary": {
            "depot": request.depot.name,
            "customers": len(request.customers),
            "deliveries": len(request.deliveries),
            "vehicles": len(request.vehicles),
            "total_boxes": total_demand,
            "total_capacity": total_capacity,
            "total_value": total_value,
            "average_speed_kmh": request.average_speed_kmh,
            "has_time_matrix": bool(request.time_matrix),
            "has_distance_matrix": bool(request.distance_matrix)
        }
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
