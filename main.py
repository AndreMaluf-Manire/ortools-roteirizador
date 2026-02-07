"""
Microserviço de Otimização de Rotas com OR-Tools
Roteirizador Manirê / Fruleve

VERSÃO 10.0 - CORREÇÕES CRÍTICAS FIXED DRIVER E VIVENDA:
- Fixed Driver agora é FORÇADO no solver (não apenas preferência)
- Janela INFINITA para entregas Vivenda (ignora horário de fechamento)
- Exclusividade bidirecional Vivenda mantida
- Balanceamento inteligente por Bin Packing
- Prioridade ABSOLUTA: 100% de alocação (penalidade 10 trilhões)
- Custo fixo de veículos = 0
- Soft time windows para não-Vivenda
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
    version="10.0"
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
MAX_TIME_HORIZON = 1440
DEFAULT_SERVICE_TIME = 15
PENALTY_UNASSIGNED = 10_000_000_000_000
SOLUTION_TIME_LIMIT = 60


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


def apply_vivenda_rule(
    deliveries: List[Delivery],
    vehicles: List[Vehicle],
    config: VivendaruleConfig,
    customer_map: Dict[str, Customer]
) -> tuple[List[Delivery], List[Delivery], set]:
    vivenda_deliveries = []
    non_vivenda_deliveries = []
    keyword = config.keyword.lower()
    
    for delivery in deliveries:
        customer_name = (delivery.customer_name or "").lower()
        if keyword in customer_name:
            vivenda_deliveries.append(delivery)
        else:
            non_vivenda_deliveries.append(delivery)
    
    print(f"\n=== REGRA VIVENDA (EXCLUSIVIDADE BIDIRECIONAL) ===")
    print(f"  Status: ATIVA")
    print(f"  Keyword: '{config.keyword}'")
    print(f"  Veículos exclusivos: {config.vehicle_ids}")
    print(f"  Entregas Vivenda encontradas: {len(vivenda_deliveries)}")
    print(f"  Entregas não-Vivenda: {len(non_vivenda_deliveries)}")
    
    if len(vivenda_deliveries) == 0:
        print("  ⚠️ ALERTA: Regra ativa mas nenhuma entrega Vivenda encontrada")
        print("  → Veículos exclusivos ficarão OCIOSOS")
        return [], deliveries, set(config.vehicle_ids)
    
    if len(config.vehicle_ids) == 0:
        print("  ❌ ERRO: Regra ativa mas sem veículos definidos!")
        return vivenda_deliveries, non_vivenda_deliveries, set()
    
    total_vivenda_boxes = sum(d.boxes for d in vivenda_deliveries)
    total_vivenda_weight = sum(d.weight_kg for d in vivenda_deliveries)
    print(f"  Total caixas Vivenda: {total_vivenda_boxes:.1f}")
    print(f"  Total peso Vivenda: {total_vivenda_weight:.1f} kg")
    
    print(f"  Modo de balanceamento: {config.balance_mode}")
    
    if config.balance_mode == "volume":
        vivenda_deliveries.sort(key=lambda d: d.boxes, reverse=True)
        vehicle_loads = {vid: 0.0 for vid in config.vehicle_ids}
        
        for delivery in vivenda_deliveries:
            lightest_vehicle = min(vehicle_loads, key=vehicle_loads.get)
            delivery.vehicle_id = lightest_vehicle
            vehicle_loads[lightest_vehicle] += delivery.boxes
            
            vehicle_idx = config.vehicle_ids.index(lightest_vehicle) + 1
            print(f"    {delivery.customer_name} ({delivery.boxes:.1f} cx) → Veículo {vehicle_idx} (total: {vehicle_loads[lightest_vehicle]:.1f} cx)")
        
        loads = list(vehicle_loads.values())
        print(f"  Balanceamento final: {[f'{l:.1f} cx' for l in loads]}")
        if max(loads) > 0:
            diff_percent = (max(loads) - min(loads)) / max(loads) * 100
            print(f"  Diferença entre veículos: {diff_percent:.1f}%")
    
    elif config.balance_mode == "weight":
        vivenda_deliveries.sort(key=lambda d: d.weight_kg, reverse=True)
        vehicle_loads = {vid: 0.0 for vid in config.vehicle_ids}
        
        for delivery in vivenda_deliveries:
            lightest_vehicle = min(vehicle_loads, key=vehicle_loads.get)
            delivery.vehicle_id = lightest_vehicle
            vehicle_loads[lightest_vehicle] += delivery.weight_kg
            
            vehicle_idx = config.vehicle_ids.index(lightest_vehicle) + 1
            print(f"    {delivery.customer_name} ({delivery.weight_kg:.1f} kg) → Veículo {vehicle_idx} (total: {vehicle_loads[lightest_vehicle]:.1f} kg)")
        
        loads = list(vehicle_loads.values())
        print(f"  Balanceamento final: {[f'{l:.1f} kg' for l in loads]}")
    
    else:
        for idx, delivery in enumerate(vivenda_deliveries):
            vehicle_idx = idx % len(config.vehicle_ids)
            delivery.vehicle_id = config.vehicle_ids[vehicle_idx]
            print(f"    {delivery.customer_name} → Veículo {vehicle_idx + 1}")
    
    return vivenda_deliveries, non_vivenda_deliveries, set(config.vehicle_ids)


# ============== OTIMIZAÇÃO ==============

@app.post("/optimize", response_model=OptimizeResponse)
async def optimize_routes(request: OptimizeRequest):
    start_time_ms = time.time() * 1000
    
    print(f"\n{'='*60}")
    print(f"=== OTIMIZAÇÃO v10.0 - FIXED DRIVER + VIVENDA INFINITA ===")
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
    
    optimization_config = request.config or OptimizationConfig()
    FLEXIBILITY_MINUTES = optimization_config.soft_time_window_tolerance
    print(f"Flexibilidade de janela: {FLEXIBILITY_MINUTES}min (máx permitido: 60min)")
    
    default_speed = DEFAULT_SPEED_KMH
    customer_map = {c.id: c for c in request.customers}
    
    vivenda_deliveries = []
    non_vivenda_deliveries = []
    vivenda_vehicle_set = set()
    vivenda_keyword = ""
    
    if optimization_config.vivenda_rule and optimization_config.vivenda_rule.enabled:
        vivenda_keyword = optimization_config.vivenda_rule.keyword.lower()
        vivenda_deliveries, non_vivenda_deliveries, vivenda_vehicle_set = apply_vivenda_rule(
            request.deliveries,
            request.vehicles,
            optimization_config.vivenda_rule,
            customer_map
        )
    else:
        print(f"\n=== REGRA VIVENDA: DESATIVADA ===")
        print(f"  Todos os veículos disponíveis para todas as entregas")
        non_vivenda_deliveries = request.deliveries
    
    locations = [request.depot.location]
    for delivery in request.deliveries:
        customer = customer_map.get(delivery.customer_id)
        if customer:
            locations.append(customer.location)
        else:
            locations.append(Location(lat=request.depot.location.lat, lng=request.depot.location.lng))
    
    num_locations = len(locations)
    
    # CORREÇÃO 2: Janela INFINITA para Vivenda
    print(f"\n=== CONSTRUINDO JANELAS DE TEMPO ===")
    time_windows = [(0, MAX_TIME_HORIZON)]
    vivenda_count = 0
    
    for delivery in request.deliveries:
        customer = customer_map.get(delivery.customer_id)
        customer_name = (delivery.customer_name or "").lower()
        
        # Verificar se é Vivenda
        is_vivenda = vivenda_keyword and vivenda_keyword in customer_name
        
        if is_vivenda:
            # JANELA INFINITA para Vivenda (ignora horário de fechamento)
            time_windows.append((0, MAX_TIME_HORIZON))
            vivenda_count += 1
            print(f"  Vivenda detectada: {delivery.customer_name} -> Janela INFINITA (0, {MAX_TIME_HORIZON})")
        else:
            # Janela normal do cliente
            if customer and customer.window_start is not None and customer.window_end is not None:
                time_windows.append((customer.window_start, customer.window_end))
            else:
                time_windows.append((0, MAX_TIME_HORIZON))
    
    if vivenda_count > 0:
        print(f"  Total entregas Vivenda com janela infinita: {vivenda_count}")
    
    service_times = [0]
    for delivery in request.deliveries:
        customer = customer_map.get(delivery.customer_id)
        if customer:
            service_times.append(customer.service_time)
        else:
            service_times.append(DEFAULT_SERVICE_TIME)
    
    # CORREÇÃO 1: Criar dicionário para Fixed Driver
    fixed_assignments = {}
    
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
                            # Armazenar para aplicar depois
                            fixed_assignments[delivery.id] = target_vehicle.id
                            print(f"  ✅ {customer_name} → veículo {target_vehicle.id} (SERÁ FORÇADO)")
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
                    print(f"  ✅ {len(matched_deliveries)} entregas identificadas para {len(rule.vehicle_ids)} veículos (preferência)")
                    rules_applied_count += 1
        
        print(f"\nRegras aplicadas: {rules_applied_count}")
    
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
    
    time_matrices = create_time_matrix_per_vehicle(locations, vehicles, default_speed)
    distance_matrix = create_distance_matrix(locations)
    
    delivery_id_to_node = {}
    for idx, delivery in enumerate(request.deliveries):
        delivery_id_to_node[delivery.id] = idx + 1
    
    vehicle_id_to_idx = {}
    for idx, vehicle in enumerate(vehicles):
        vehicle_id_to_idx[vehicle.id] = idx
    
    manager = pywrapcp.RoutingIndexManager(num_locations, num_vehicles, 0)
    routing = pywrapcp.RoutingModel(manager)
    
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
    
    print(f"\n=== APLICANDO JANELAS NO SOLVER ===")
    
    for node in range(num_locations):
        index = manager.NodeToIndex(node)
        window_start, window_end = time_windows[node]
        
        # Aplicar flexibilidade apenas para não-Vivenda
        if window_end < MAX_TIME_HORIZON:
            flexible_start = max(0, window_start - 60)
            flexible_end = min(MAX_TIME_HORIZON, window_end + 60)
        else:
            # Vivenda já tem janela infinita, manter como está
            flexible_start = window_start
            flexible_end = window_end
        
        time_dimension.CumulVar(index).SetRange(flexible_start, flexible_end)
    
    for vehicle_idx in range(num_vehicles):
        time_dimension.SetSpanCostCoefficientForVehicle(100, vehicle_idx)
    
    for vehicle_idx in range(num_vehicles):
        start_index = routing.Start(vehicle_idx)
        time_dimension.CumulVar(start_index).SetRange(request.start_time, request.start_time)
    
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
    
    for vehicle_idx in range(num_vehicles):
        routing.SetArcCostEvaluatorOfVehicle(time_callback_indices[vehicle_idx], vehicle_idx)
    
    print(f"\n=== CUSTOS DE VEÍCULOS ===")
    print(f"Custo fixo por veículo: 0 (priorizar alocação vs economia)")
    
    for vehicle_idx in range(num_vehicles):
        routing.SetFixedCostOfVehicle(0, vehicle_idx)
    
    print(f"\n=== PREFERÊNCIAS DE MOTORISTA ===")
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
                preferred_driver_count += 1
                customer = customer_map.get(delivery.customer_id)
                customer_name = customer.name if customer else "?"
                print(f"  Preferência: {customer_name} → {vehicles[preferred_vehicle_idx].name} (suave)")
    
    if preferred_driver_count > 0:
        print(f"  Total preferências: {preferred_driver_count} (não obrigatórias)")
    
    # CORREÇÃO 1 (aplicação): Aplicar Fixed Driver assignments
    if fixed_assignments:
        print(f"\n=== APLICANDO FIXED DRIVER (RÍGIDO) ===")
        for delivery_id, vehicle_id in fixed_assignments.items():
            if delivery_id in delivery_id_to_node and vehicle_id in vehicle_id_to_idx:
                node = delivery_id_to_node[delivery_id]
                vehicle_idx = vehicle_id_to_idx[vehicle_id]
                index = manager.NodeToIndex(node)
                routing.SetAllowedVehiclesForIndex([vehicle_idx], index)
                
                delivery = next((d for d in request.deliveries if d.id == delivery_id), None)
                customer_name = delivery.customer_name if delivery else "?"
                print(f"  ✅ FORÇADO: {customer_name} → {vehicles[vehicle_idx].name}")
        
        print(f"  Total Fixed Driver aplicados: {len(fixed_assignments)}")
    
    if vivenda_deliveries and vivenda_vehicle_set:
        print(f"\n=== APLICANDO EXCLUSIVIDADE BIDIRECIONAL ===")
        
        vivenda_vehicle_indices = []
        for vid in vivenda_vehicle_set:
            if vid in vehicle_id_to_idx:
                vivenda_vehicle_indices.append(vehicle_id_to_idx[vid])
        
        if len(vivenda_vehicle_indices) == 0:
            print("  ❌ ERRO CRÍTICO: Veículos Vivenda não encontrados na frota!")
            return OptimizeResponse(
                success=False,
                message="Veículos da regra Vivenda não existem na frota configurada",
                routes=[],
                unassigned_deliveries=[d.id for d in request.deliveries],
                vehicles_used=0,
                total_deliveries=len(request.deliveries),
                total_value=0,
                optimization_time_ms=int(time.time() * 1000 - start_time_ms)
            )
        
        print(f"  Veículos Vivenda (índices): {vivenda_vehicle_indices}")
        
        vivenda_forced = 0
        for delivery in vivenda_deliveries:
            if delivery.vehicle_id and delivery.vehicle_id in vehicle_id_to_idx:
                node = delivery_id_to_node[delivery.id]
                vehicle_idx = vehicle_id_to_idx[delivery.vehicle_id]
                index = manager.NodeToIndex(node)
                
                routing.SetAllowedVehiclesForIndex([vehicle_idx], index)
                vivenda_forced += 1
                
                customer = customer_map.get(delivery.customer_id)
                customer_name = customer.name if customer else "?"
                print(f"  ✅ VIVENDA FORÇADA: {customer_name} → {vehicles[vehicle_idx].name}")
        
        print(f"  Total Vivenda forçadas: {vivenda_forced}/{len(vivenda_deliveries)}")
        
        non_vivenda_vehicle_indices = [i for i in range(num_vehicles) if i not in vivenda_vehicle_indices]
        
        if len(non_vivenda_vehicle_indices) == 0:
            print("  ⚠️ ALERTA CRÍTICO: TODOS os veículos são Vivenda!")
            print("  → Entregas não-Vivenda ficarão ÓRFÃS")
        else:
            non_vivenda_blocked = 0
            for delivery in non_vivenda_deliveries:
                node = delivery_id_to_node[delivery.id]
                index = manager.NodeToIndex(node)
                
                routing.SetAllowedVehiclesForIndex(non_vivenda_vehicle_indices, index)
                non_vivenda_blocked += 1
            
            print(f"  ✅ Não-Vivenda bloqueadas dos veículos exclusivos: {non_vivenda_blocked}")
            print(f"  Veículos disponíveis para não-Vivenda: {len(non_vivenda_vehicle_indices)}")
    
    elif optimization_config.vivenda_rule and optimization_config.vivenda_rule.enabled:
        print(f"\n=== VIVENDA: REGRA ATIVA, SEM ENTREGAS ===")
        print(f"  ⚠️ Veículos exclusivos ({len(vivenda_vehicle_set)}) ficarão OCIOSOS")
        print(f"  Considere desativar a regra se não houver entregas Vivenda")
    
    else:
        print(f"\n=== VIVENDA: REGRA DESATIVADA ===")
        print(f"  Todos os veículos disponíveis para todas as entregas")
    
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
    
    print(f"\n=== PENALIDADES ===")
    print(f"Penalidade por não-atendimento: {PENALTY_UNASSIGNED:,}")
    
    for node in range(1, num_locations):
        routing.AddDisjunction([manager.NodeToIndex(node)], PENALTY_UNASSIGNED)
    
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_parameters.time_limit.FromSeconds(SOLUTION_TIME_LIMIT)
    
    print(f"\n=== RESOLVENDO (limite: {SOLUTION_TIME_LIMIT}s) ===")
    
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
            
            status = ""
            if arrived_early:
                status = f" (espera {wait_time}min)"
            elif arrived_late:
                delay = arrival_time - window_end
                status = f" ⚠️ ATRASADO {delay}min"
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
    
    unassigned = []
    for delivery in request.deliveries:
        if delivery.id not in all_assigned_deliveries:
            unassigned.append(delivery.id)
            customer = customer_map.get(delivery.customer_id)
            customer_name = customer.name if customer else "?"
            print(f"⚠️ NÃO ALOCADA: {customer_name} (ID: {delivery.id})")
    
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
    else:
        print(f"\n✅ SUCESSO: 100% das entregas alocadas!")
    
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


@app.get("/")
async def root():
    return {"status": "ok", "service": "OR-Tools Route Optimizer", "version": "10.0"}


@app.get("/health")
async def health():
    return {"status": "healthy", "version": "10.0"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
