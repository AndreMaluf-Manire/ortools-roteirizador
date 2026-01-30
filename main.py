"""
Microserviço de Otimização de Rotas com OR-Tools
Roteirizador Manirê / Fruleve

VERSÃO 7.6.0 - GARANTIA DE 100% DE ALOCAÇÃO:
- Penalidade EXTREMA para não-atendimento (10 trilhões)
- Tempo de solução aumentado para 300s (5 minutos)
- Prioridade absoluta: alocar TODAS as entregas
- Usar apenas veículos reais (sem veículos extras fantasma)
- Soft time windows mais flexíveis (30min)
- Logs otimizados para evitar rate limit do Railway

VERSÃO 7.5.0 - MOTORISTA PREFERENCIAL:
- preferred_driver_id: Motorista preferencial do cliente (cadastro)
- Pré-atribuição automática ao primeiro veículo do motorista

VERSÃO 7.4.0 - NOVAS REGRAS DE ROTEAMENTO:
- vehicle_exclusive_for_group: Veículo(s) exclusivo(s) para grupo específico
- force_multiple_vehicles: Forçar distribuição em múltiplos veículos
- fixed_driver com múltiplos veículos (vehicle_ids em action)
"""

from fastapi import FastAPI, HTTPException, Header
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
    version="7.6.0"
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

# v7.6.0: Penalidade EXTREMA para garantir 100% de alocação
PENALTY_UNASSIGNED = 10_000_000_000_000  # 10 trilhões
SOLUTION_TIME_LIMIT = 300  # 5 minutos
FLEXIBILITY_MINUTES = 30  # Flexibilidade de janela aumentada


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


class OptimizeRequest(BaseModel):
    depot: Depot
    customers: List[Customer]
    deliveries: List[Delivery]
    vehicles: List[Vehicle]
    start_time: int = 360
    mode: str = "minimize_vehicles"
    delivery_groups: Optional[List[List[str]]] = None
    routing_rules: Optional[List[RoutingRule]] = None


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
                dist = haversine_distance(
                    locations[i].lat, locations[i].lng,
                    locations[j].lat, locations[j].lng
                )
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
                matrix[i][j] = haversine_distance(
                    locations[i].lat, locations[i].lng,
                    locations[j].lat, locations[j].lng
                )
    
    return matrix


# ============== OTIMIZAÇÃO ==============

@app.post("/optimize", response_model=OptimizeResponse)
async def optimize_routes(request: OptimizeRequest, x_api_key: str = Header(None)):
    start_time_ms = time.time() * 1000
    
    # Validar API key
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    print(f"\n{'='*60}")
    print(f"=== OTIMIZAÇÃO v7.6.0 - GARANTIA 100% ALOCAÇÃO ===")
    print(f"{'='*60}")
    print(f"Depot: {request.depot.name}")
    print(f"Entregas: {len(request.deliveries)}")
    print(f"Veículos: {len(request.vehicles)}")
    print(f"Modo: {request.mode}")
    print(f"Penalidade não-atendimento: {PENALTY_UNASSIGNED:,}")
    print(f"Tempo limite solução: {SOLUTION_TIME_LIMIT}s")
    
    if len(request.deliveries) == 0:
        return OptimizeResponse(
            success=True,
            message="Nenhuma entrega para otimizar",
            routes=[],
            unassigned_deliveries=[],
            vehicles_used=0,
            total_deliveries=0,
            total_value=0,
            optimization_time_ms=int(time.time() * 1000 - start_time_ms)
        )
    
    # Velocidade padrão
    default_speed = DEFAULT_SPEED_KMH
    
    # Criar mapa de clientes
    customer_map = {c.id: c for c in request.customers}
    
    # Preparar localizações
    locations = [request.depot.location]
    for delivery in request.deliveries:
        customer = customer_map.get(delivery.customer_id)
        if customer:
            locations.append(customer.location)
        else:
            locations.append(Location(lat=request.depot.location.lat, lng=request.depot.location.lng))
    
    num_locations = len(locations)
    
    # Preparar janelas de tempo
    time_windows = [(0, MAX_TIME_HORIZON)]
    for delivery in request.deliveries:
        customer = customer_map.get(delivery.customer_id)
        if customer and customer.window_start is not None and customer.window_end is not None:
            time_windows.append((customer.window_start, customer.window_end))
        else:
            time_windows.append((0, MAX_TIME_HORIZON))
    
    # Preparar tempos de serviço
    service_times = [0]
    for delivery in request.deliveries:
        customer = customer_map.get(delivery.customer_id)
        if customer:
            service_times.append(customer.service_time)
        else:
            service_times.append(DEFAULT_SERVICE_TIME)
    
    # ===== PROCESSAR REGRAS DE ROTEAMENTO =====
    rules_applied_count = 0
    if request.routing_rules:
        print(f"\n=== PROCESSANDO REGRAS DE ROTEAMENTO ===")
        print(f"Total de regras recebidas: {len(request.routing_rules)}")
        
        sorted_rules = sorted(request.routing_rules, key=lambda r: r.priority, reverse=True)
        
        for rule in sorted_rules:
            print(f"\nAplicando regra: {rule.type} (prioridade: {rule.priority})")
            
            if rule.type == "fixed_driver":
                if not rule.action or not rule.action.freteiro_id:
                    print(f"  ⚠️ Regra sem freteiro_id, pulando")
                    continue
                
                freteiro_id = rule.action.freteiro_id
                target_vehicle_ids = rule.action.vehicle_ids if rule.action.vehicle_ids else None
                
                print(f"  Condição: {rule.condition.field} {rule.condition.operator} '{rule.condition.value}'")
                
                for delivery in request.deliveries:
                    if delivery.vehicle_id:
                        continue
                    
                    customer_name = delivery.customer_name or ""
                    matches = False
                    
                    if rule.condition.operator == "contains":
                        matches = rule.condition.value.lower() in customer_name.lower()
                    elif rule.condition.operator == "starts_with":
                        matches = customer_name.lower().startswith(rule.condition.value.lower())
                    elif rule.condition.operator == "equals":
                        matches = customer_name.lower() == rule.condition.value.lower()
                    
                    if matches:
                        target_vehicle = None
                        for v in request.vehicles:
                            if v.freteiro_id == freteiro_id:
                                if target_vehicle_ids is None or v.id in target_vehicle_ids:
                                    target_vehicle = v
                                    break
                        
                        if target_vehicle:
                            delivery.vehicle_id = target_vehicle.id
                            print(f"  ✅ {customer_name} → veículo {target_vehicle.id}")
                            rules_applied_count += 1
            
            elif rule.type == "max_stops":
                if rule.action and rule.action.max_stops:
                    print(f"  Máximo de paradas: {rule.action.max_stops}")
                    rules_applied_count += 1
            
            elif rule.type == "group_by_name":
                if not rule.condition:
                    continue
                
                group_deliveries = []
                for delivery in request.deliveries:
                    customer_name = delivery.customer_name or ""
                    matches = False
                    
                    if rule.condition.operator == "contains":
                        matches = rule.condition.value.lower() in customer_name.lower()
                    elif rule.condition.operator == "starts_with":
                        matches = customer_name.lower().startswith(rule.condition.value.lower())
                    elif rule.condition.operator == "equals":
                        matches = customer_name.lower() == rule.condition.value.lower()
                    
                    if matches:
                        group_deliveries.append(delivery.id)
                
                if len(group_deliveries) > 1:
                    if not request.delivery_groups:
                        request.delivery_groups = []
                    request.delivery_groups.append(group_deliveries)
                    print(f"  ✅ Agrupadas {len(group_deliveries)} entregas")
                    rules_applied_count += 1
            
            elif rule.type == "vehicle_exclusive_for_group":
                if not rule.vehicle_ids or len(rule.vehicle_ids) == 0:
                    continue
                
                print(f"  Veículos exclusivos: {len(rule.vehicle_ids)} veículos")
                
                allowed_delivery_ids = set()
                for delivery in request.deliveries:
                    customer_name = delivery.customer_name or ""
                    matches = False
                    
                    if rule.condition and rule.condition.operator == "contains":
                        matches = rule.condition.value.lower() in customer_name.lower()
                    elif rule.condition and rule.condition.operator == "starts_with":
                        matches = customer_name.lower().startswith(rule.condition.value.lower())
                    elif rule.condition and rule.condition.operator == "equals":
                        matches = customer_name.lower() == rule.condition.value.lower()
                    
                    if matches:
                        allowed_delivery_ids.add(delivery.id)
                
                print(f"  Entregas permitidas: {len(allowed_delivery_ids)}")
                
                if not hasattr(request, 'vehicle_exclusive_rules'):
                    request.vehicle_exclusive_rules = []
                request.vehicle_exclusive_rules.append({
                    'vehicle_ids': rule.vehicle_ids,
                    'allowed_delivery_ids': allowed_delivery_ids
                })
                rules_applied_count += 1
            
            elif rule.type == "force_multiple_vehicles":
                if not rule.vehicle_ids or len(rule.vehicle_ids) < 2:
                    continue
                
                matched_deliveries = []
                for delivery in request.deliveries:
                    customer_name = delivery.customer_name or ""
                    matches = False
                    
                    if rule.condition and rule.condition.operator == "contains":
                        matches = rule.condition.value.lower() in customer_name.lower()
                    elif rule.condition and rule.condition.operator == "starts_with":
                        matches = customer_name.lower().startswith(rule.condition.value.lower())
                    elif rule.condition and rule.condition.operator == "equals":
                        matches = customer_name.lower() == rule.condition.value.lower()
                    
                    if matches:
                        matched_deliveries.append(delivery)
                
                if len(matched_deliveries) >= 2:
                    distribution_mode = rule.action.distribution_mode if rule.action and rule.action.distribution_mode else "balanced"
                    
                    for idx, delivery in enumerate(matched_deliveries):
                        if not delivery.vehicle_id:
                            vehicle_idx = idx % len(rule.vehicle_ids)
                            delivery.vehicle_id = rule.vehicle_ids[vehicle_idx]
                    
                    print(f"  ✅ Distribuídas {len(matched_deliveries)} entregas em {len(rule.vehicle_ids)} veículos")
                    rules_applied_count += 1
        
        print(f"\nRegras aplicadas: {rules_applied_count}")
    
    # ===== CONFIGURAR VEÍCULOS =====
    # v7.6.0: USAR APENAS VEÍCULOS REAIS - sem veículos extras!
    num_vehicles = len(request.vehicles)
    vehicles = list(request.vehicles)
    
    print(f"\n=== VEÍCULOS ===")
    print(f"Total de veículos REAIS: {num_vehicles}")
    
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
    
    for idx, v in enumerate(vehicles):
        speed = v.average_speed_kmh if v.average_speed_kmh else default_speed
        print(f"  {idx+1}. {v.name}: {v.capacity_boxes} caixas, {speed} km/h")
    
    # Criar matrizes de tempo por veículo
    time_matrices = create_time_matrix_per_vehicle(locations, vehicles, default_speed)
    distance_matrix = create_distance_matrix(locations)
    
    # Mapear delivery_id para node_index
    delivery_id_to_node = {}
    for idx, delivery in enumerate(request.deliveries):
        delivery_id_to_node[delivery.id] = idx + 1
    
    # Mapear vehicle_id para vehicle_idx
    vehicle_id_to_idx = {}
    for idx, vehicle in enumerate(vehicles):
        vehicle_id_to_idx[vehicle.id] = idx
    
    # ===== CRIAR MODELO OR-TOOLS =====
    manager = pywrapcp.RoutingIndexManager(num_locations, num_vehicles, 0)
    routing = pywrapcp.RoutingModel(manager)
    
    # ----- CALLBACKS DE TEMPO POR VEÍCULO -----
    def make_time_callback(vehicle_idx):
        time_matrix = time_matrices.get(vehicle_idx, time_matrices[0])
        def time_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            travel = time_matrix[from_node][to_node]
            service = service_times[to_node] if to_node > 0 else 0
            return travel + service
        return time_callback
    
    time_callback_indices = []
    for vehicle_idx in range(num_vehicles):
        callback = make_time_callback(vehicle_idx)
        callback_index = routing.RegisterTransitCallback(callback)
        time_callback_indices.append(callback_index)
    
    # ----- DIMENSÃO DE TEMPO COM TIME WINDOWS -----
    base_time_matrix = time_matrices.get(0, create_time_matrix(locations, default_speed))
    
    def base_time_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        travel = base_time_matrix[from_node][to_node]
        service = service_times[to_node] if to_node > 0 else 0
        return travel + service
    
    base_time_callback_index = routing.RegisterTransitCallback(base_time_callback)
    
    routing.AddDimension(
        base_time_callback_index,
        MAX_TIME_HORIZON,
        MAX_TIME_HORIZON,
        False,
        "Time"
    )
    time_dimension = routing.GetDimensionOrDie("Time")
    
    # v7.6.0: Janelas de tempo mais flexíveis (30min)
    print(f"\n=== JANELAS DE TEMPO (flexibilidade: {FLEXIBILITY_MINUTES}min) ===")
    for node in range(num_locations):
        index = manager.NodeToIndex(node)
        window_start, window_end = time_windows[node]
        
        flexible_start = max(0, window_start - FLEXIBILITY_MINUTES)
        flexible_end = min(MAX_TIME_HORIZON, window_end + FLEXIBILITY_MINUTES)
        
        time_dimension.CumulVar(index).SetRange(flexible_start, flexible_end)
    
    # Penalidade por atraso
    for vehicle_idx in range(num_vehicles):
        time_dimension.SetSpanCostCoefficientForVehicle(5000, vehicle_idx)
    
    # Horário de início
    for vehicle_idx in range(num_vehicles):
        start_index = routing.Start(vehicle_idx)
        time_dimension.CumulVar(start_index).SetRange(request.start_time, request.start_time)
    
    # ----- CALLBACK DE CAPACIDADE -----
    def demand_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        if from_node == 0:
            return 0
        delivery = request.deliveries[from_node - 1]
        return int(math.ceil(delivery.boxes))
    
    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,
        [v.capacity_boxes for v in vehicles],
        True,
        "Capacity"
    )
    
    # ----- CUSTOS POR VEÍCULO -----
    for vehicle_idx in range(num_vehicles):
        routing.SetArcCostEvaluatorOfVehicle(time_callback_indices[vehicle_idx], vehicle_idx)
    
    # v7.6.0: Custo fixo MENOR para não penalizar uso de mais veículos
    # Prioridade é alocar TODAS as entregas, não minimizar veículos
    for vehicle_idx in range(num_vehicles):
        routing.SetFixedCostOfVehicle(1000, vehicle_idx)  # Custo baixo
    
    # ----- PRÉ-ATRIBUIÇÃO POR MOTORISTA PREFERENCIAL -----
    print(f"\n=== PRÉ-ATRIBUIÇÕES ===")
    preferred_driver_count = 0
    for delivery in request.deliveries:
        if delivery.vehicle_id:
            continue
        
        if delivery.preferred_driver_id:
            preferred_vehicle_idx = None
            for idx, vehicle in enumerate(vehicles):
                if vehicle.freteiro_id == delivery.preferred_driver_id:
                    preferred_vehicle_idx = idx
                    break
            
            if preferred_vehicle_idx is not None:
                node = delivery_id_to_node[delivery.id]
                index = manager.NodeToIndex(node)
                routing.SetAllowedVehiclesForIndex([preferred_vehicle_idx], index)
                preferred_driver_count += 1
                customer = customer_map.get(delivery.customer_id)
                customer_name = customer.name if customer else "?"
                print(f"  Motorista Preferencial: {customer_name} -> {vehicles[preferred_vehicle_idx].name}")
    
    if preferred_driver_count > 0:
        print(f"  Total por motorista preferencial: {preferred_driver_count}")
    
    # ----- PRÉ-ATRIBUIÇÃO POR VEHICLE_ID -----
    pre_assigned_count = 0
    for delivery in request.deliveries:
        if delivery.vehicle_id and delivery.vehicle_id in vehicle_id_to_idx:
            node = delivery_id_to_node[delivery.id]
            vehicle_idx = vehicle_id_to_idx[delivery.vehicle_id]
            index = manager.NodeToIndex(node)
            routing.SetAllowedVehiclesForIndex([vehicle_idx], index)
            pre_assigned_count += 1
            customer = customer_map.get(delivery.customer_id)
            customer_name = customer.name if customer else "?"
            print(f"  Pré-atribuição: {customer_name} -> {vehicles[vehicle_idx].name}")
    
    if pre_assigned_count > 0:
        print(f"  Total pré-atribuições: {pre_assigned_count}")
    
    # ----- GRUPOS DE ENTREGAS -----
    if request.delivery_groups:
        print(f"\n=== GRUPOS DE ENTREGAS ===")
        for group_idx, group in enumerate(request.delivery_groups):
            if len(group) < 2:
                continue
            
            group_nodes = []
            for delivery_id in group:
                if delivery_id in delivery_id_to_node:
                    group_nodes.append(delivery_id_to_node[delivery_id])
            
            if len(group_nodes) >= 2:
                first_node = group_nodes[0]
                first_index = manager.NodeToIndex(first_node)
                
                for node in group_nodes[1:]:
                    index = manager.NodeToIndex(node)
                    routing.solver().Add(
                        routing.VehicleVar(first_index) == routing.VehicleVar(index)
                    )
                
                print(f"  Grupo {group_idx + 1}: {len(group_nodes)} entregas vinculadas")
    
    # ----- RESTRIÇÕES DE EXCLUSIVIDADE -----
    if hasattr(request, 'vehicle_exclusive_rules') and request.vehicle_exclusive_rules:
        print(f"\n=== RESTRIÇÕES DE EXCLUSIVIDADE ===")
        for rule_data in request.vehicle_exclusive_rules:
            vehicle_ids = rule_data['vehicle_ids']
            allowed_delivery_ids = rule_data['allowed_delivery_ids']
            
            exclusive_vehicle_indices = []
            for vid in vehicle_ids:
                if vid in vehicle_id_to_idx:
                    exclusive_vehicle_indices.append(vehicle_id_to_idx[vid])
            
            if not exclusive_vehicle_indices:
                continue
            
            for delivery in request.deliveries:
                node = delivery_id_to_node[delivery.id]
                index = manager.NodeToIndex(node)
                
                if delivery.vehicle_id:
                    continue
                
                if delivery.id in allowed_delivery_ids:
                    routing.SetAllowedVehiclesForIndex(exclusive_vehicle_indices, index)
                else:
                    allowed_vehicles = [i for i in range(num_vehicles) if i not in exclusive_vehicle_indices]
                    if allowed_vehicles:
                        routing.SetAllowedVehiclesForIndex(allowed_vehicles, index)
    
    # ----- PENALIDADES v7.6.0 -----
    # Penalidade EXTREMA para garantir 100% de alocação
    print(f"\n=== PENALIDADES ===")
    print(f"Penalidade por não-atendimento: {PENALTY_UNASSIGNED:,}")
    
    for node in range(1, num_locations):
        routing.AddDisjunction([manager.NodeToIndex(node)], PENALTY_UNASSIGNED)
    
    # ----- PARÂMETROS DE BUSCA v7.6.0 -----
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_parameters.time_limit.FromSeconds(SOLUTION_TIME_LIMIT)
    
    print(f"\n=== RESOLVENDO (limite: {SOLUTION_TIME_LIMIT}s) ===")
    
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
    
    print("✅ Solução encontrada!")
    
    # ===== EXTRAIR SOLUÇÃO =====
    routes = []
    vehicles_used = 0
    total_value_all = 0
    all_assigned_deliveries = set()
    
    for vehicle_idx in range(num_vehicles):
        index = routing.Start(vehicle_idx)
        
        route_nodes = []
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            if node > 0:
                route_nodes.append(node)
            index = solution.Value(routing.NextVar(index))
        
        if not route_nodes:
            continue
        
        vehicle = vehicles[vehicle_idx]
        vehicle_speed = vehicle.average_speed_kmh if vehicle.average_speed_kmh else default_speed
        time_matrix = time_matrices.get(vehicle_idx, base_time_matrix)
        
        stops = []
        route_boxes = 0.0
        route_weight = 0.0
        route_value = 0.0
        route_distance = 0.0
        route_wait_time = 0
        route_service_time = 0
        route_travel_time = 0
        
        current_time = request.start_time
        prev_node = 0
        
        print(f"\n=== ROTA {vehicle_idx + 1}: {vehicle.name} ({vehicle_speed} km/h) ===")
        print(f"Saída do depot: {minutes_to_time(request.start_time)}")
        
        for seq, node in enumerate(route_nodes):
            delivery = request.deliveries[node - 1]
            customer = customer_map.get(delivery.customer_id)
            
            all_assigned_deliveries.add(delivery.id)
            
            travel_time = time_matrix[prev_node][node]
            distance = distance_matrix[prev_node][node]
            
            arrival_time = current_time + travel_time
            
            window_start, window_end = time_windows[node]
            
            wait_time = 0
            arrived_early = False
            arrived_late = False
            
            if arrival_time < window_start:
                wait_time = window_start - arrival_time
                arrived_early = True
                service_start = window_start
            else:
                service_start = arrival_time
                if arrival_time > window_end:
                    arrived_late = True
            
            service_time = service_times[node]
            departure_time = service_start + service_time
            
            route_boxes += delivery.boxes
            route_weight += delivery.weight_kg
            route_value += delivery.value
            route_distance += distance
            route_wait_time += wait_time
            route_service_time += service_time
            route_travel_time += travel_time
            
            stop = RouteStop(
                delivery_id=delivery.id,
                customer_id=delivery.customer_id,
                customer_name=customer.name if customer else "?",
                location=locations[node],
                sequence=seq + 1,
                arrival_time=arrival_time,
                departure_time=departure_time,
                service_time=service_time,
                window_start=window_start,
                window_end=window_end,
                travel_time=travel_time,
                distance_km=round(distance, 1),
                wait_time=wait_time,
                arrived_early=arrived_early,
                arrived_late=arrived_late,
                boxes=delivery.boxes,
                weight_kg=delivery.weight_kg,
                value=delivery.value
            )
            stops.append(stop)
            
            # Log resumido para evitar rate limit
            status = ""
            if arrived_early:
                status = f" (espera {wait_time}min)"
            elif arrived_late:
                status = " ⚠️ ATRASADO"
            print(f"  {seq+1}. {customer.name if customer else '?'}: {minutes_to_time(arrival_time)}-{minutes_to_time(departure_time)}{status}")
            
            current_time = departure_time
            prev_node = node
        
        total_time = current_time - request.start_time
        
        route = Route(
            vehicle_id=vehicle.id,
            vehicle_name=vehicle.name,
            stops=stops,
            total_boxes=round(route_boxes, 1),
            total_weight=round(route_weight, 1),
            total_value=round(route_value, 2),
            total_distance_km=round(route_distance, 1),
            total_time_minutes=total_time,
            start_time=request.start_time,
            end_time=current_time
        )
        routes.append(route)
        vehicles_used += 1
        total_value_all += route_value
        
        print(f"  RESUMO: {len(stops)} entregas, {round(route_boxes, 1)} caixas, {round(route_distance, 1)} km, {total_time} min")
    
    # Identificar entregas não alocadas
    unassigned = []
    for delivery in request.deliveries:
        if delivery.id not in all_assigned_deliveries:
            unassigned.append(delivery.id)
            customer = customer_map.get(delivery.customer_id)
            customer_name = customer.name if customer else "?"
            print(f"⚠️ NÃO ALOCADA: {customer_name}")
    
    optimization_time = int(time.time() * 1000 - start_time_ms)
    
    print(f"\n{'='*60}")
    print(f"=== RESUMO FINAL ===")
    print(f"{'='*60}")
    print(f"Veículos usados: {vehicles_used}/{num_vehicles}")
    print(f"Entregas alocadas: {len(all_assigned_deliveries)}/{len(request.deliveries)}")
    print(f"Entregas pendentes: {len(unassigned)}")
    print(f"Tempo de otimização: {optimization_time}ms")
    
    if len(unassigned) > 0:
        print(f"\n⚠️ ATENÇÃO: {len(unassigned)} entregas NÃO foram alocadas!")
        print(f"Possíveis causas: capacidade insuficiente, janelas incompatíveis, restrições de veículos")
    
    return OptimizeResponse(
        success=True,
        message=f"Otimização concluída. {vehicles_used} veículos, {len(all_assigned_deliveries)} entregas alocadas, {len(unassigned)} pendentes.",
        routes=routes,
        unassigned_deliveries=unassigned,
        vehicles_used=vehicles_used,
        total_deliveries=len(request.deliveries),
        total_value=round(total_value_all, 2),
        optimization_time_ms=optimization_time
    )


# ============== ENDPOINTS ==============

@app.get("/")
async def root():
    return {"status": "ok", "service": "OR-Tools Route Optimizer", "version": "7.6.0"}


@app.get("/health")
async def health():
    return {"status": "healthy", "version": "7.6.0"}
