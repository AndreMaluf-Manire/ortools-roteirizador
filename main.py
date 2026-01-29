"""
Microserviço de Otimização de Rotas com OR-Tools
Roteirizador Manirê / Fruleve

VERSÃO 5.0 - CORREÇÃO CRÍTICA:
- Propagação SEQUENCIAL de tempo: arrival[i] = departure[i-1] + travel_time
- OR-Tools define apenas a ORDEM dos pontos
- Tempos são recalculados DEPOIS de forma propagada
- effective_start = max(arrival, window_start)
- departure = effective_start + service_time
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
    version="5.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    window_start: Optional[int] = None
    window_end: Optional[int] = None
    service_time: int = 15


class Delivery(BaseModel):
    id: str
    customer_id: str
    boxes: float = 0  # Mudado para float para suportar 0.5
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
    start_time: int = 360
    max_route_duration: int = 720
    mode: str = "minimize_vehicles"
    average_speed_kmh: float = 16.0


class Stop(BaseModel):
    delivery_id: str
    customer_id: str
    customer_name: str
    stop_order: int
    arrival_time: int           # Quando o carro CHEGA fisicamente
    effective_start: int        # Quando COMEÇA a entregar (= max(arrival, window_start))
    departure_time: int         # Quando o carro SAI (= effective_start + service_time)
    service_time: int
    wait_time: int = 0          # Tempo esperando (= effective_start - arrival)
    travel_time_from_prev: int = 0
    distance_from_prev_km: float = 0
    window_start: Optional[int]
    window_end: Optional[int]
    window_ok: bool
    arrived_early: bool = False
    arrived_late: bool = False
    boxes: float  # Mudado para float
    weight_kg: float
    value: float = 0


class Route(BaseModel):
    vehicle_id: Optional[str]
    vehicle_name: Optional[str]
    stops: List[Stop]
    total_boxes: float  # Mudado para float
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
    R = 6371
    lat1, lon1 = math.radians(loc1.lat), math.radians(loc1.lng)
    lat2, lon2 = math.radians(loc2.lat), math.radians(loc2.lng)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    return R * c


def get_location_for_node(node: int, request: 'OptimizeRequest', customer_map: dict) -> Location:
    if node == 0:
        return request.depot.location
    else:
        delivery = request.deliveries[node - 1]
        customer = customer_map.get(delivery.customer_id)
        return customer.location if customer else request.depot.location


def calculate_travel_time(loc1: Location, loc2: Location, speed_kmh: float) -> int:
    """Calcula tempo de viagem em minutos baseado na distância e velocidade média"""
    distance_km = haversine_distance(loc1, loc2)
    if speed_kmh <= 0:
        speed_kmh = 16.0
    travel_time_minutes = (distance_km / speed_kmh) * 60
    return max(1, int(travel_time_minutes))


def minutes_to_time(minutes: int) -> str:
    hours = minutes // 60
    mins = minutes % 60
    return f"{hours:02d}:{mins:02d}"


# ============== SOLVER OR-TOOLS ==============

def solve_vrptw(request: OptimizeRequest) -> OptimizeResponse:
    """
    VERSÃO 5.0 - PROPAGAÇÃO SEQUENCIAL DE TEMPO
    
    1. OR-Tools define a ORDEM dos pontos (otimização)
    2. DEPOIS recalculamos os tempos de forma PROPAGADA:
       - arrival[i] = departure[i-1] + travel_time[i-1 → i]
       - effective_start[i] = max(arrival[i], window_start[i])
       - departure[i] = effective_start[i] + service_time[i]
    """
    start_time_ms = time.time() * 1000
    
    customer_map = {c.id: c for c in request.customers}
    delivery_map = {d.id: d for d in request.deliveries}
    
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
    
    # Callback de distância (para otimização de ordem)
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        loc1 = get_location_for_node(from_node, request, customer_map)
        loc2 = get_location_for_node(to_node, request, customer_map)
        # Retorna distância em metros (inteiro)
        return int(haversine_distance(loc1, loc2) * 1000)
    
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    
    # Dimensão de capacidade
    def demand_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        if from_node == 0:
            return 0
        delivery = request.deliveries[from_node - 1]
        return int(delivery.boxes)  # OR-Tools precisa de int
    
    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,
        [v.capacity_boxes for v in vehicles],
        True,
        "Capacity"
    )
    
    # Penalidades
    if request.mode == "minimize_vehicles":
        for vehicle_idx in range(num_vehicles):
            routing.SetFixedCostOfVehicle(10000, vehicle_idx)
    
    penalty = 1000000
    for node in range(1, num_locations):
        routing.AddDisjunction([manager.NodeToIndex(node)], penalty)
    
    # Parâmetros de busca
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_parameters.time_limit.FromSeconds(30)
    
    # Resolver
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
    
    # ===== EXTRAIR SOLUÇÃO E RECALCULAR TEMPOS (VERSÃO 5.0) =====
    routes = []
    vehicles_used = 0
    total_value_all = 0
    
    for vehicle_idx in range(num_vehicles):
        index = routing.Start(vehicle_idx)
        
        # Coletar sequência de nós (apenas a ORDEM)
        route_nodes = []
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            if node > 0:
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
        
        # Começar do depot no horário de início
        # current_departure = horário que o veículo SAI do ponto atual
        current_departure = request.start_time
        prev_location = request.depot.location
        
        print(f"\n=== ROTA {vehicle_idx + 1} ===")
        print(f"Saída do depot: {minutes_to_time(current_departure)}")
        
        for stop_order, node in enumerate(route_nodes, 1):
            delivery = request.deliveries[node - 1]
            customer = customer_map.get(delivery.customer_id)
            customer_location = customer.location if customer else request.depot.location
            
            # 1. Calcular tempo de viagem do ponto anterior
            travel_time = calculate_travel_time(prev_location, customer_location, request.average_speed_kmh)
            distance = haversine_distance(prev_location, customer_location)
            
            # 2. Calcular horário de CHEGADA física
            # arrival = departure do ponto anterior + tempo de viagem
            arrival_time = current_departure + travel_time
            
            # 3. Tempo de serviço deste cliente
            service_time = customer.service_time if customer else 15
            
            # 4. Calcular effective_start (quando COMEÇA a entregar)
            # Se chegou antes de abrir, espera até abrir
            window_start = customer.window_start if customer else None
            window_end = customer.window_end if customer else None
            
            if window_start is not None and arrival_time < window_start:
                # Chegou ANTES de abrir - ESPERA
                effective_start = window_start
                wait_time = window_start - arrival_time
                arrived_early = True
            else:
                # Chegou depois de abrir (ou sem janela)
                effective_start = arrival_time
                wait_time = 0
                arrived_early = False
            
            # 5. Verificar se chegou ATRASADO (depois de fechar)
            arrived_late = False
            if window_end is not None and arrival_time > window_end:
                arrived_late = True
            
            # 6. Calcular horário de SAÍDA (departure)
            # departure = effective_start + service_time
            departure_time = effective_start + service_time
            
            # 7. window_ok = não está atrasado
            window_ok = not arrived_late
            
            # Log para debug
            customer_name = customer.name if customer else "Desconhecido"
            print(f"\nPonto {stop_order}: {customer_name}")
            print(f"  Travel time: {travel_time}min ({distance:.1f}km)")
            print(f"  Chegada física: {minutes_to_time(arrival_time)}")
            if window_start:
                print(f"  Janela: {minutes_to_time(window_start)} - {minutes_to_time(window_end) if window_end else '??'}")
            if arrived_early:
                print(f"  ADIANTADO! Espera {wait_time}min até abrir")
            if arrived_late:
                print(f"  ATRASADO! Chegou depois de fechar")
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
                distance_from_prev_km=round(distance, 2),
                window_start=window_start,
                window_end=window_end,
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
            
            # ===== ATUALIZAR PARA O PRÓXIMO PONTO =====
            # O próximo ponto começa a partir do DEPARTURE deste!
            current_departure = departure_time
            prev_location = customer_location
        
        # Adicionar distância de volta ao depot
        distance_to_depot = haversine_distance(prev_location, request.depot.location)
        route_distance += distance_to_depot
        
        # Tempo total da rota
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
            total_boxes=round(route_boxes, 1),
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
    return {"status": "ok", "service": "OR-Tools Route Optimizer", "version": "5.0.0"}


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "ortools_version": "9.x",
        "api_key_configured": API_KEY != "dev-key-change-in-production",
        "version": "5.0.0",
        "features": [
            "SEQUENTIAL time propagation (v5.0)",
            "arrival = prev_departure + travel_time",
            "effective_start = max(arrival, window_start)",
            "departure = effective_start + service_time",
            "configurable average_speed_kmh (default 16)",
            "service_time per customer",
            "wait_time when arrived early",
            "real distance calculation",
            "arrived_early/arrived_late flags",
            "capacity_used_percent",
            "total_value in R$",
            "decimal boxes support (0.5)"
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
