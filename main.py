"""
Microserviço de Otimização de Rotas com OR-Tools
Roteirizador Manirê / Fruleve

VERSÃO 6.0 - ROBUSTA:
- Time Windows como HARD CONSTRAINT no OR-Tools
- Dimensão de TEMPO (não distância) para otimização
- Service time incluído na dimensão de tempo
- Propagação sequencial de tempos após solução
- Janelas de horário RESPEITADAS na otimização
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
    version="6.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY = os.getenv("ORTOOLS_API_KEY", "dev-key-change-in-production")

# Constantes
DEFAULT_SPEED_KMH = 16.0
MAX_TIME_HORIZON = 1440  # 24 horas em minutos
DEFAULT_SERVICE_TIME = 15


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
    window_start: Optional[int] = None  # minutos desde meia-noite
    window_end: Optional[int] = None
    service_time: int = DEFAULT_SERVICE_TIME


class Delivery(BaseModel):
    id: str
    customer_id: str
    boxes: float = 0
    weight_kg: float = 0
    value: float = 0


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
    time_matrix: Optional[List[List[int]]] = None
    distance_matrix: Optional[List[List[float]]] = None
    start_time: int = 360  # 06:00 por padrão
    max_route_duration: int = 720  # 12 horas
    mode: str = "minimize_vehicles"
    average_speed_kmh: float = DEFAULT_SPEED_KMH


class Stop(BaseModel):
    delivery_id: str
    customer_id: str
    customer_name: str
    stop_order: int
    arrival_time: int
    effective_start: int
    departure_time: int
    service_time: int
    wait_time: int = 0
    travel_time_from_prev: int = 0
    distance_from_prev_km: float = 0
    window_start: Optional[int]
    window_end: Optional[int]
    window_ok: bool
    arrived_early: bool = False
    arrived_late: bool = False
    boxes: float
    weight_kg: float
    value: float = 0


class Route(BaseModel):
    vehicle_id: Optional[str]
    vehicle_name: Optional[str]
    stops: List[Stop]
    total_boxes: float
    total_weight_kg: float
    total_value: float = 0
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
    total_value: float = 0
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


def calculate_travel_time_minutes(distance_km: float, speed_kmh: float) -> int:
    """Calcula tempo de viagem em minutos"""
    if speed_kmh <= 0:
        speed_kmh = DEFAULT_SPEED_KMH
    travel_time = (distance_km / speed_kmh) * 60
    return max(1, int(round(travel_time)))


def minutes_to_time(minutes: int) -> str:
    """Converte minutos desde meia-noite para formato HH:MM"""
    hours = minutes // 60
    mins = minutes % 60
    return f"{hours:02d}:{mins:02d}"


def create_time_matrix(locations: List[Location], speed_kmh: float) -> List[List[int]]:
    """Cria matriz de tempo de viagem entre todos os pontos"""
    n = len(locations)
    matrix = []
    for i in range(n):
        row = []
        for j in range(n):
            if i == j:
                row.append(0)
            else:
                dist = haversine_distance(locations[i], locations[j])
                time_min = calculate_travel_time_minutes(dist, speed_kmh)
                row.append(time_min)
        matrix.append(row)
    return matrix


def create_distance_matrix(locations: List[Location]) -> List[List[float]]:
    """Cria matriz de distância em km entre todos os pontos"""
    n = len(locations)
    matrix = []
    for i in range(n):
        row = []
        for j in range(n):
            if i == j:
                row.append(0.0)
            else:
                dist = haversine_distance(locations[i], locations[j])
                row.append(round(dist, 2))
        matrix.append(row)
    return matrix


# ============== SOLVER OR-TOOLS ==============

def solve_vrptw(request: OptimizeRequest) -> OptimizeResponse:
    """
    VERSÃO 6.0 - VRPTW com Time Windows como HARD CONSTRAINT
    
    1. Cria dimensão de TEMPO com service_time
    2. Adiciona Time Windows como restrição
    3. OR-Tools otimiza respeitando janelas
    4. Recalcula tempos propagados para exibição
    """
    start_time_ms = time.time() * 1000
    
    # Mapas de lookup
    customer_map = {c.id: c for c in request.customers}
    delivery_map = {d.id: d for d in request.deliveries}
    
    # Nós: depot (0) + entregas (1, 2, 3, ...)
    num_locations = 1 + len(request.deliveries)
    
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
    
    # Construir lista de localizações na ordem dos nós
    locations = [request.depot.location]
    for delivery in request.deliveries:
        customer = customer_map.get(delivery.customer_id)
        if customer:
            locations.append(customer.location)
        else:
            locations.append(request.depot.location)
    
    # Criar matrizes de tempo e distância
    speed = request.average_speed_kmh if request.average_speed_kmh > 0 else DEFAULT_SPEED_KMH
    time_matrix = create_time_matrix(locations, speed)
    distance_matrix = create_distance_matrix(locations)
    
    # Service times por nó (depot = 0)
    service_times = [0]  # Depot não tem service time
    for delivery in request.deliveries:
        customer = customer_map.get(delivery.customer_id)
        st = customer.service_time if customer and customer.service_time else DEFAULT_SERVICE_TIME
        service_times.append(st)
    
    # Time windows por nó
    # Depot: pode operar o dia todo (ou usar start_time como início)
    time_windows = [(request.start_time, MAX_TIME_HORIZON)]
    
    for delivery in request.deliveries:
        customer = customer_map.get(delivery.customer_id)
        if customer and customer.window_start is not None and customer.window_end is not None:
            # Janela definida
            time_windows.append((customer.window_start, customer.window_end))
        else:
            # Sem janela: pode entregar a qualquer hora
            time_windows.append((0, MAX_TIME_HORIZON))
    
    print(f"\n=== CONFIGURAÇÃO ===")
    print(f"Velocidade média: {speed} km/h")
    print(f"Horário início: {minutes_to_time(request.start_time)}")
    print(f"Entregas: {len(request.deliveries)}")
    print(f"Veículos disponíveis: {len(request.vehicles)}")
    
    # Configurar veículos
    if request.mode == "minimize_vehicles":
        # Criar veículos suficientes (até o número de entregas)
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
        vehicles = list(request.vehicles)
    
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
    
    # ===== CRIAR MODELO OR-TOOLS =====
    manager = pywrapcp.RoutingIndexManager(num_locations, num_vehicles, 0)
    routing = pywrapcp.RoutingModel(manager)
    
    # ----- CALLBACK DE TEMPO (inclui service time) -----
    def time_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        # Tempo = viagem + service time no destino
        travel = time_matrix[from_node][to_node]
        service = service_times[to_node] if to_node > 0 else 0
        return travel + service
    
    time_callback_index = routing.RegisterTransitCallback(time_callback)
    
    # ----- DIMENSÃO DE TEMPO COM TIME WINDOWS -----
    routing.AddDimension(
        time_callback_index,
        MAX_TIME_HORIZON,  # Slack máximo (permite espera)
        MAX_TIME_HORIZON,  # Tempo máximo por veículo
        False,  # Não força início em 0
        "Time"
    )
    time_dimension = routing.GetDimensionOrDie("Time")
    
    # Definir janelas de tempo para cada nó
    for node in range(num_locations):
        index = manager.NodeToIndex(node)
        window_start, window_end = time_windows[node]
        time_dimension.CumulVar(index).SetRange(window_start, window_end)
        
        if node > 0:
            delivery = request.deliveries[node - 1]
            customer = customer_map.get(delivery.customer_id)
            customer_name = customer.name if customer else "?"
            print(f"  Nó {node} ({customer_name}): janela {minutes_to_time(window_start)}-{minutes_to_time(window_end)}")
    
    # Definir horário de início dos veículos
    for vehicle_idx in range(num_vehicles):
        start_index = routing.Start(vehicle_idx)
        time_dimension.CumulVar(start_index).SetRange(request.start_time, request.start_time)
    
    # ----- CALLBACK DE CAPACIDADE -----
    def demand_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        if from_node == 0:
            return 0
        delivery = request.deliveries[from_node - 1]
        return int(math.ceil(delivery.boxes))  # Arredondar para cima
    
    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # Sem slack
        [v.capacity_boxes for v in vehicles],
        True,  # Começa em 0
        "Capacity"
    )
    
    # ----- PENALIDADES E CUSTOS -----
    # Custo por arco = tempo de viagem (otimiza por tempo total)
    routing.SetArcCostEvaluatorOfAllVehicles(time_callback_index)
    
    # Custo fixo por veículo (para minimizar número de veículos)
    if request.mode == "minimize_vehicles":
        for vehicle_idx in range(num_vehicles):
            routing.SetFixedCostOfVehicle(100000, vehicle_idx)
    
    # Penalidade alta para entregas não atendidas
    penalty = 10000000  # 10 milhões
    for node in range(1, num_locations):
        routing.AddDisjunction([manager.NodeToIndex(node)], penalty)
    
    # ----- PARÂMETROS DE BUSCA -----
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_parameters.time_limit.FromSeconds(60)  # Mais tempo para soluções melhores
    
    print(f"\n=== RESOLVENDO ===")
    
    # ----- RESOLVER -----
    solution = routing.SolveWithParameters(search_parameters)
    
    if not solution:
        print("ERRO: Não encontrou solução!")
        return OptimizeResponse(
            success=False,
            message="Não foi possível encontrar uma solução. Verifique janelas de tempo e capacidades.",
            routes=[],
            unassigned_deliveries=[d.id for d in request.deliveries],
            vehicles_used=0,
            total_deliveries=len(request.deliveries),
            total_value=0,
            optimization_time_ms=int(time.time() * 1000 - start_time_ms)
        )
    
    print("Solução encontrada!")
    
    # ===== EXTRAIR SOLUÇÃO E RECALCULAR TEMPOS =====
    routes = []
    vehicles_used = 0
    total_value_all = 0
    
    for vehicle_idx in range(num_vehicles):
        index = routing.Start(vehicle_idx)
        
        # Coletar nós da rota
        route_nodes = []
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            if node > 0:  # Ignorar depot
                route_nodes.append(node)
            index = solution.Value(routing.NextVar(index))
        
        if not route_nodes:
            continue
        
        # ===== RECALCULAR TEMPOS DE FORMA PROPAGADA =====
        stops = []
        route_boxes = 0.0
        route_weight = 0.0
        route_value = 0.0
        route_distance = 0.0
        route_wait_time = 0
        route_service_time = 0
        route_travel_time = 0
        
        current_time = request.start_time  # Horário atual (departure do ponto anterior)
        prev_node = 0  # Começa no depot
        
        vehicle = vehicles[vehicle_idx] if vehicle_idx < len(vehicles) else None
        print(f"\n=== ROTA {vehicle_idx + 1}: {vehicle.name if vehicle else 'Extra'} ===")
        print(f"Saída do depot: {minutes_to_time(current_time)}")
        
        for stop_order, node in enumerate(route_nodes, 1):
            delivery = request.deliveries[node - 1]
            customer = customer_map.get(delivery.customer_id)
            customer_name = customer.name if customer else "Desconhecido"
            
            # Tempo de viagem do ponto anterior
            travel_time = time_matrix[prev_node][node]
            distance = distance_matrix[prev_node][node]
            
            # Horário de chegada física
            arrival_time = current_time + travel_time
            
            # Service time deste cliente
            service_time = service_times[node]
            
            # Janela de tempo
            window_start = time_windows[node][0]
            window_end = time_windows[node][1]
            
            # Calcular effective_start (quando COMEÇA a entregar)
            if arrival_time < window_start:
                # Chegou ANTES de abrir - ESPERA
                effective_start = window_start
                wait_time = window_start - arrival_time
                arrived_early = True
            else:
                effective_start = arrival_time
                wait_time = 0
                arrived_early = False
            
            # Verificar se chegou ATRASADO
            arrived_late = arrival_time > window_end
            window_ok = not arrived_late
            
            # Horário de saída
            departure_time = effective_start + service_time
            
            # Log
            print(f"\nPonto {stop_order}: {customer_name}")
            print(f"  Viagem: {travel_time}min ({distance:.1f}km)")
            print(f"  Chegada física: {minutes_to_time(arrival_time)}")
            print(f"  Janela: {minutes_to_time(window_start)}-{minutes_to_time(window_end)}")
            if arrived_early:
                print(f"  ⏰ ADIANTADO! Espera {wait_time}min")
            if arrived_late:
                print(f"  ⚠️ ATRASADO!")
            print(f"  Início atendimento: {minutes_to_time(effective_start)}")
            print(f"  Service time: {service_time}min")
            print(f"  Saída: {minutes_to_time(departure_time)}")
            
            stops.append(Stop(
                delivery_id=delivery.id,
                customer_id=delivery.customer_id,
                customer_name=customer_name,
                stop_order=stop_order,
                arrival_time=arrival_time,
                effective_start=effective_start,
                departure_time=departure_time,
                service_time=service_time,
                wait_time=wait_time,
                travel_time_from_prev=travel_time,
                distance_from_prev_km=distance,
                window_start=window_start if window_start > 0 else None,
                window_end=window_end if window_end < MAX_TIME_HORIZON else None,
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
            
            # Atualizar para próximo ponto
            current_time = departure_time
            prev_node = node
        
        # Distância de volta ao depot
        distance_to_depot = distance_matrix[prev_node][0]
        route_distance += distance_to_depot
        
        # Tempo total
        total_time = stops[-1].departure_time - request.start_time if stops else 0
        
        # Ocupação
        capacity_used_percent = 0
        if vehicle and vehicle.capacity_boxes > 0:
            capacity_used_percent = round((route_boxes / vehicle.capacity_boxes) * 100, 1)
        
        vehicles_used += 1
        total_value_all += route_value
        
        routes.append(Route(
            vehicle_id=vehicle.id if vehicle and not vehicle.id.startswith("extra_") else None,
            vehicle_name=vehicle.name if vehicle else f"Rota Extra {vehicle_idx + 1}",
            stops=stops,
            total_boxes=round(route_boxes, 2),
            total_weight_kg=round(route_weight, 2),
            total_value=round(route_value, 2),
            total_time_minutes=total_time,
            total_distance_km=round(route_distance, 2),
            total_wait_time=route_wait_time,
            total_service_time=route_service_time,
            total_travel_time=route_travel_time,
            capacity_used_percent=capacity_used_percent
        ))
    
    # Entregas não alocadas
    assigned = set()
    for route in routes:
        for stop in route.stops:
            assigned.add(stop.delivery_id)
    
    unassigned = [d.id for d in request.deliveries if d.id not in assigned]
    
    if unassigned:
        print(f"\n⚠️ {len(unassigned)} entregas não alocadas!")
    
    return OptimizeResponse(
        success=True,
        message=f"Otimização concluída: {vehicles_used} veículos, {len(assigned)} entregas alocadas",
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
    return {"status": "ok", "service": "OR-Tools Route Optimizer", "version": "6.0.0"}


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "ortools_version": "9.x",
        "api_key_configured": API_KEY != "dev-key-change-in-production",
        "version": "6.0.0",
        "features": [
            "Time Windows as HARD CONSTRAINT (v6.0)",
            "Time dimension with service_time",
            "Optimizes by total time (not distance)",
            "Sequential time propagation",
            "Wait time when arrived early",
            "Configurable average_speed_kmh (default 16)",
            "High penalty for unassigned (10M)",
            "60 second solver timeout",
            "Decimal boxes support"
        ]
    }


@app.post("/optimize", response_model=OptimizeResponse)
async def optimize_routes(
    request: OptimizeRequest,
    authorization: Optional[str] = Header(None)
):
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
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Erro na otimização: {str(e)}")


@app.post("/validate")
async def validate_request(request: OptimizeRequest):
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
            "average_speed_kmh": request.average_speed_kmh
        }
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
