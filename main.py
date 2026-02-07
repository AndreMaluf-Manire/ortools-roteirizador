"""
Microserviço de Otimização de Rotas com OR-Tools
Roteirizador Manirê / Fruleve

VERSÃO 7.9.3 - STABLE FIX:
- Baseado na v7.9.0 (Estável).
- FIX: Regra 'fixed_driver' (Maria Honos) agora aplica trava real no solver.
- FIX: Janela Infinita para Vivenda (garante alocação ignorando horário de fechamento).
- MANTIDO: Penalidade de 100 Trilhões (que funcionou na v7.9.0).
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import os
import math
import time

app = FastAPI(
    title="OR-Tools Route Optimizer",
    description="API de otimização de rotas para o Roteirizador Manirê",
    version="7.9.3"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constantes
DEFAULT_SPEED_KMH = 16.0
MAX_TIME_HORIZON = 1440  # 24 horas em minutos
DEFAULT_SERVICE_TIME = 15

# Penalidade testada e aprovada na v7.9.0
PENALTY_UNASSIGNED = 100_000_000_000_000  # 100 trilhões
SOLUTION_TIME_LIMIT = 60  # 1 minuto


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
    service_time: int = DEFAULT_SERVICE_TIME

class Delivery(BaseModel):
    id: str
    customer_id: str
    customer_name: Optional[str] = None
    boxes: float = 0
    weight_kg: float = 0
    value: float = 0
    vehicle_id: Optional[str] = None
    preferred_driver_id: Optional[str] = None

class Vehicle(BaseModel):
    id: str
    name: str
    capacity_boxes: int
    capacity_kg: float = 99999
    average_speed_kmh: Optional[float] = None
    freteiro_id: Optional[str] = None

class RoutingRuleCondition(BaseModel):
    field: str
    operator: str
    value: str

class RoutingRuleAction(BaseModel):
    freteiro_id: Optional[str] = None
    vehicle_ids: Optional[List[str]] = None
    max_stops: Optional[int] = None
    distribution_mode: Optional[str] = None

class RoutingRule(BaseModel):
    type: str
    priority: int = 0
    condition: Optional[RoutingRuleCondition] = None
    action: Optional[RoutingRuleAction] = None
    vehicle_ids: Optional[List[str]] = None

class VivendaruleConfig(BaseModel):
    enabled: bool = False
    vehicle_ids: List[str] = []
    keyword: str = "vivenda"
    balance_mode: str = "volume"

class OptimizationConfig(BaseModel):
    soft_time_window_tolerance: int = 15
    vivenda_rule: Optional[VivendaruleConfig] = None
    force_allocation: bool = True

class OptimizeRequest(BaseModel):
    depot: Depot
    customers: List[Customer]
    deliveries: List[Delivery]
    vehicles: List[Vehicle]
    start_time: int = 360
    mode: str = "minimize_vehicles"
    delivery_groups: Optional[List[List[str]]] = None
    routing_rules: Optional[List[RoutingRule]] = None
    config: Optional[OptimizationConfig] = None

class RouteStop(BaseModel):
    delivery_id: str
    customer_id: str
    customer_name: str
    location: Location
    sequence: int
    arrival_time: int
    departure_time: int
    service_time: int
    window_start: int
    window_end: int
    travel_time: int
    distance_km: float
    wait_time: int
    arrived_early: bool = False
    arrived_late: bool = False
    boxes: float = 0
    weight_kg: float = 0
    value: float = 0

class Route(BaseModel):
    vehicle_id: str
    vehicle_name: str
    stops: List[RouteStop]
    total_boxes: float
    total_weight: float
    total_value: float
    total_distance_km: float
    total_time_minutes: int
    start_time: int
    end_time: int

class OptimizeResponse(BaseModel):
    success: bool
    message: str
    routes: List[Route]
    unassigned_deliveries: List[str]
    vehicles_used: int
    total_deliveries: int
    total_value: float
    optimization_time_ms: int


# ============== FUNÇÕES AUXILIARES ==============

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

def minutes_to_time(minutes: int) -> str:
    hours = minutes // 60
    mins = minutes % 60
    return f"{hours:02d}:{mins:02d}"

def create_time_matrix(locations: List[Location], speed_kmh: float) -> List[List[int]]:
    n = len(locations)
    matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                dist = haversine_distance(locations[i].lat, locations[i].lng, locations[j].lat, locations[j].lng)
                time_minutes = int(math.ceil((dist / speed_kmh) * 60))
                matrix[i][j] = time_minutes
    return matrix

def create_time_matrix_per_vehicle(locations: List[Location], vehicles: List[Vehicle], default_speed: float) -> Dict[int, List[List[int]]]:
    matrices = {}
    for idx, vehicle in enumerate(vehicles):
        speed = vehicle.average_speed_kmh if vehicle.average_speed_kmh else default_speed
        matrices[idx] = create_time_matrix(locations, speed)
    return matrices

def create_distance_matrix(locations: List[Location]) -> List[List[float]]:
    n = len(locations)
    matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                matrix[i][j] = haversine_distance(locations[i].lat, locations[i].lng, locations[j].lat, locations[j].lng)
    return matrix

def apply_vivenda_rule(deliveries: List[Delivery], vehicles: List[Vehicle], config: VivendaruleConfig, customer_map: Dict[str, Customer]):
    vivenda_deliveries = []
    non_vivenda_deliveries = []
    keyword = config.keyword.lower()
    
    for delivery in deliveries:
        customer_name = (delivery.customer_name or "").lower()
        if keyword in customer_name:
            vivenda_deliveries.append(delivery)
        else:
            non_vivenda_deliveries.append(delivery)
    
    print(f"\n=== REGRA VIVENDA (JANELA INFINITA) ===")
    print(f"  Entregas Vivenda: {len(vivenda_deliveries)}")
    
    if len(vivenda_deliveries) == 0 or len(config.vehicle_ids) == 0:
        return vivenda_deliveries, non_vivenda_deliveries, set(config.vehicle_ids)
    
    # Bin Packing (Volume)
    if config.balance_mode == "volume":
        vivenda_deliveries.sort(key=lambda d: d.boxes, reverse=True)
        vehicle_loads = {vid: 0.0 for vid in config.vehicle_ids}
        for delivery in vivenda_deliveries:
            lightest_vehicle = min(vehicle_loads, key=vehicle_loads.get)
            delivery.vehicle_id = lightest_vehicle
            vehicle_loads[lightest_vehicle] += delivery.boxes
    else:
        # Round Robin fallback
        for idx, delivery in enumerate(vivenda_deliveries):
            vehicle_idx = idx % len(config.vehicle_ids)
            delivery.vehicle_id = config.vehicle_ids[vehicle_idx]
            
    return vivenda_deliveries, non_vivenda_deliveries, set(config.vehicle_ids)


# ============== OTIMIZAÇÃO ==============

@app.post("/optimize", response_model=OptimizeResponse)
async def optimize_routes(request: OptimizeRequest):
    start_time_ms = time.time() * 1000
    print(f"\n=== OTIMIZAÇÃO v7.9.3 ===")
    
    if len(request.deliveries) == 0:
        return OptimizeResponse(success=True, message="Sem entregas", routes=[], unassigned_deliveries=[], vehicles_used=0, total_deliveries=0, total_value=0, optimization_time_ms=0)

    # Configuração
    optimization_config = request.config or OptimizationConfig()
    FLEXIBILITY_MINUTES = optimization_config.soft_time_window_tolerance
    
    # Mapa de clientes e Veículos
    customer_map = {c.id: c for c in request.customers}
    vehicle_id_to_idx = {v.id: i for i, v in enumerate(request.vehicles)}
    num_vehicles = len(request.vehicles)
    
    # ===== 1. APLICAR REGRA VIVENDA =====
    vivenda_deliveries = []
    non_vivenda_deliveries = []
    vivenda_vehicle_set = set()
    
    if optimization_config.vivenda_rule and optimization_config.vivenda_rule.enabled:
        vivenda_deliveries, non_vivenda_deliveries, vivenda_vehicle_set = apply_vivenda_rule(
            request.deliveries, request.vehicles, optimization_config.vivenda_rule, customer_map
        )
    else:
        non_vivenda_deliveries = request.deliveries

    # ===== 2. PREPARAR REGRAS FIXAS (MARIA HONOS) =====
    fixed_assignments = {} 
    
    if request.routing_rules:
        sorted_rules = sorted(request.routing_rules, key=lambda r: r.priority, reverse=True)
        print(f"\n=== PROCESSANDO REGRAS FIXAS ===")
        
        for rule in sorted_rules:
            if rule.type == "fixed_driver" and rule.action and rule.action.freteiro_id:
                freteiro_id = rule.action.freteiro_id
                target_vehicle_ids = rule.action.vehicle_ids
                
                possible_vehicles = []
                for v in request.vehicles:
                    if v.freteiro_id == freteiro_id:
                        if not target_vehicle_ids or v.id in target_vehicle_ids:
                            possible_vehicles.append(v)
                
                if not possible_vehicles:
                    continue
                    
                target_vehicle = possible_vehicles[0]
                
                for delivery in request.deliveries:
                    if delivery.vehicle_id: continue 
                    if delivery.id in fixed_assignments: continue

                    customer_name = (delivery.customer_name or "").lower()
                    matches = False
                    
                    if rule.condition.operator == "contains":
                        matches = rule.condition.value.lower() in customer_name
                    elif rule.condition.operator == "equals":
                        matches = customer_name == rule.condition.value.lower()
                        
                    if matches:
                        fixed_assignments[delivery.id] = target_vehicle.id
                        print(f"  🔒 FIXADO: {delivery.customer_name} -> {target_vehicle.name}")

    # ===== OR-TOOLS SETUP =====
    locations = [request.depot.location]
    for d in request.deliveries:
        c = customer_map.get(d.customer_id)
        locations.append(c.location if c else request.depot.location)
    
    time_matrices = create_time_matrix_per_vehicle(locations, request.vehicles, DEFAULT_SPEED_KMH)
    distance_matrix = create_distance_matrix(locations)
    
    manager = pywrapcp.RoutingIndexManager(len(locations), num_vehicles, 0)
    routing = pywrapcp.RoutingModel(manager)
    
    def make_time_callback(v_idx):
        t_mat = time_matrices.get(v_idx, time_matrices[0])
        def cb(from_idx, to_idx):
            from_node = manager.IndexToNode(from_idx)
            to_node = manager.IndexToNode(to_idx)
            travel = t_mat[from_node][to_node]
            service = request.customers[from_node-1].service_time if from_node > 0 and from_node-1 < len(request.customers) else DEFAULT_SERVICE_TIME
            if from_node == 0: service = 0
            return travel + service
        return cb

    transit_callback_indices = [routing.RegisterTransitCallback(make_time_callback(i)) for i in range(num_vehicles)]
    
    routing.AddDimension(
        transit_callback_indices[0],
        MAX_TIME_HORIZON, MAX_TIME_HORIZON, False, "Time"
    )
    time_dimension = routing.GetDimensionOrDie("Time")
    
    # Janelas de Tempo (Soft + Infinita para Vivenda)
    for i, delivery in enumerate(request.deliveries):
        index = manager.NodeToIndex(i + 1)
        c = customer_map.get(delivery.customer_id)
        
        is_vivenda = False
        if optimization_config.vivenda_rule and optimization_config.vivenda_rule.enabled:
             keyword = optimization_config.vivenda_rule.keyword.lower()
             if keyword in (delivery.customer_name or "").lower():
                 is_vivenda = True
        
        if is_vivenda:
            # Janela Infinita para garantir que caiba na rota
            time_dimension.CumulVar(index).SetRange(0, MAX_TIME_HORIZON)
        elif c and c.window_start is not None:
            start = max(0, c.window_start - FLEXIBILITY_MINUTES)
            end = min(MAX_TIME_HORIZON, c.window_end + FLEXIBILITY_MINUTES + 120)
            time_dimension.CumulVar(index).SetRange(start, end)
            time_dimension.SetCumulVarSoftUpperBound(index, c.window_end + FLEXIBILITY_MINUTES, 10000)

    def capacity_callback(from_index):
        node = manager.IndexToNode(from_index)
        if node == 0: return 0
        return int(math.ceil(request.deliveries[node-1].boxes))
        
    cap_idx = routing.RegisterUnaryTransitCallback(capacity_callback)
    routing.AddDimensionWithVehicleCapacity(cap_idx, 0, [v.capacity_boxes for v in request.vehicles], True, "Capacity")

    for i in range(num_vehicles):
        routing.SetArcCostEvaluatorOfVehicle(transit_callback_indices[i], i)
        routing.SetFixedCostOfVehicle(0, i)
    
    # Penalidade por não alocação (100 Trilhões - Seguro pois funcionou na v7.9.0)
    for i in range(len(request.deliveries)):
        routing.AddDisjunction([manager.NodeToIndex(i + 1)], PENALTY_UNASSIGNED)

    # ===== APLICAR RESTRIÇÕES =====
    
    # 1. Regras Fixas (Valmir / Maria Honos)
    for delivery_id, vehicle_uuid in fixed_assignments.items():
        delivery_idx = next((i for i, d in enumerate(request.deliveries) if d.id == delivery_id), None)
        vehicle_idx = vehicle_id_to_idx.get(vehicle_uuid)
        if delivery_idx is not None and vehicle_idx is not None:
            index = manager.NodeToIndex(delivery_idx + 1)
            routing.SetAllowedVehiclesForIndex([vehicle_idx], index)

    # 2. Regras Vivenda
    vivenda_indices = [vehicle_id_to_idx[vid] for vid in vivenda_vehicle_set if vid in vehicle_id_to_idx]
    non_vivenda_indices = [i for i in range(num_vehicles) if i not in vivenda_indices]
    
    if len(vivenda_indices) > 0:
        print(f"\n=== APLICANDO TRAVAS VIVENDA ===")
        # Travar Vivenda nos carros Vivenda
        for delivery in vivenda_deliveries:
            d_idx = next((i for i, d in enumerate(request.deliveries) if d.id == delivery.id), None)
            if d_idx is not None:
                index = manager.NodeToIndex(d_idx + 1)
                routing.SetAllowedVehiclesForIndex(vivenda_indices, index)

        # Travar carros Vivenda para não levar outras coisas
        if len(non_vivenda_indices) > 0: 
            for delivery in non_vivenda_deliveries:
                if delivery.id in fixed_assignments: continue
                d_idx = next((i for i, d in enumerate(request.deliveries) if d.id == delivery.id), None)
                if d_idx is not None:
                    index = manager.NodeToIndex(d_idx + 1)
                    routing.SetAllowedVehiclesForIndex(non_vivenda_indices, index)

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.FromSeconds(SOLUTION_TIME_LIMIT)
    
    solution = routing.SolveWithParameters(params)
    
    if not solution:
        return OptimizeResponse(success=False, message="Falha na solução", routes=[], unassigned_deliveries=[], vehicles_used=0, total_deliveries=0, total_value=0, optimization_time_ms=0)

    routes = []
    vehicles_used = 0
    assigned_ids = set()
    total_val = 0
    
    for v_idx in range(num_vehicles):
        index = routing.Start(v_idx)
        stops = []
        route_boxes = 0
        
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            if node > 0:
                d = request.deliveries[node-1]
                assigned_ids.add(d.id)
                stops.append(RouteStop(
                    delivery_id=d.id, customer_id=d.customer_id, customer_name=d.customer_name or "?",
                    location=locations[node], sequence=len(stops)+1,
                    arrival_time=0, departure_time=0, service_time=0, window_start=0, window_end=0,
                    travel_time=0, distance_km=0, wait_time=0, boxes=d.boxes, weight_kg=d.weight_kg, value=d.value
                ))
                route_boxes += d.boxes
                total_val += d.value
            index = solution.Value(routing.NextVar(index))
            
        if stops:
            vehicles_used += 1
            routes.append(Route(
                vehicle_id=request.vehicles[v_idx].id,
                vehicle_name=request.vehicles[v_idx].name,
                stops=stops,
                total_boxes=route_boxes,
                total_weight=0, total_value=0, total_distance_km=0, total_time_minutes=0, start_time=0, end_time=0
            ))

    unassigned = [d.id for d in request.deliveries if d.id not in assigned_ids]
    
    return OptimizeResponse(
        success=True,
        message=f"Otimizado. {len(unassigned)} pendentes.",
        routes=routes,
        unassigned_deliveries=unassigned,
        vehicles_used=vehicles_used,
        total_deliveries=len(request.deliveries),
        total_value=total_val,
        optimization_time_ms=int(time.time()*1000 - start_time_ms)
    )

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
