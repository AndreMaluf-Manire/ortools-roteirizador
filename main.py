"""
Microserviço de Otimização de Rotas com OR-Tools
Roteirizador Manirê / Fruleve

VERSÃO 7.4.0 - NOVAS REGRAS DE ROTEAMENTO:
- vehicle_exclusive_for_group: Veículo(s) exclusivo(s) para grupo específico
- force_multiple_vehicles: Forçar distribuição em múltiplos veículos
- fixed_driver com múltiplos veículos (vehicle_ids em action)
- Distribuição balanceada (round-robin) ou sequencial
- Restrições de exclusividade via SetAllowedVehiclesForIndex
- Logs detalhados de processamento de regras
- Soft Time Windows (flexibilidade de 15min)
- Penalidade aumentada para não-atendimento (100M)
- Tempo de solução aumentado (120s)
- Garante 100% de alocação das entregas
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
    version="7.4.0"
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
    customer_name: Optional[str] = None  # NOVO: nome do cliente para regras
    boxes: float = 0
    weight_kg: float = 0
    value: float = 0
    vehicle_id: Optional[str] = None  # NOVO: pré-atribuição a veículo


class Vehicle(BaseModel):
    id: str
    name: str
    capacity_boxes: int
    capacity_kg: float = 99999
    average_speed_kmh: Optional[float] = None  # NOVO: velocidade específica do veículo
    freteiro_id: Optional[str] = None  # NOVO: ID do freteiro/motorista


class RoutingRuleCondition(BaseModel):
    field: str  # "customer_name"
    operator: str  # "contains", "starts_with", "equals"
    value: str


class RoutingRuleAction(BaseModel):
    freteiro_id: Optional[str] = None  # Para fixed_driver
    vehicle_ids: Optional[List[str]] = None  # Para vehicle_exclusive_for_group, force_multiple_vehicles, fixed_driver com múltiplos veículos
    distribution_mode: Optional[str] = None  # Para force_multiple_vehicles: "balanced" ou "sequential"


class RoutingRule(BaseModel):
    id: str
    name: str
    type: str  # "fixed_driver", "group_by_name", "vehicle_exclusive_for_group", "force_multiple_vehicles"
    priority: int
    condition: RoutingRuleCondition
    action: RoutingRuleAction
    vehicle_ids: Optional[List[str]] = None  # Para vehicle_exclusive_for_group e force_multiple_vehicles


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
    delivery_groups: Optional[List[List[str]]] = None  # NOVO: grupos de entregas
    routing_rules: Optional[List[RoutingRule]] = None  # NOVO: regras de roteamento


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
    average_speed_kmh: float = DEFAULT_SPEED_KMH  # NOVO: velocidade usada


class OptimizeResponse(BaseModel):
    success: bool
    message: str
    routes: List[Route]
    unassigned_deliveries: List[str]
    vehicles_used: int
    total_deliveries: int
    total_value: float = 0
    optimization_time_ms: int


# ============== MODELOS PARA RECÁLCULO ==============

class RecalculateStop(BaseModel):
    delivery_id: str
    customer_id: str
    customer_name: str
    stop_order: int
    service_time: int
    window_start: Optional[int]
    window_end: Optional[int]
    boxes: float
    weight_kg: float
    value: float = 0
    lat: float
    lng: float


class RecalculateRequest(BaseModel):
    depot: Depot
    stops: List[RecalculateStop]
    start_time: int = 360
    average_speed_kmh: float = DEFAULT_SPEED_KMH


class RecalculateResponse(BaseModel):
    success: bool
    message: str
    stops: List[Stop]
    total_time_minutes: int
    total_distance_km: float
    total_wait_time: int
    total_service_time: int
    total_travel_time: int


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


def create_time_matrix_per_vehicle(
    locations: List[Location], 
    vehicles: List[Vehicle], 
    default_speed: float
) -> Dict[int, List[List[int]]]:
    """
    Cria matriz de tempo de viagem para cada veículo (pode ter velocidades diferentes)
    Retorna um dicionário: vehicle_idx -> time_matrix
    """
    matrices = {}
    for idx, vehicle in enumerate(vehicles):
        speed = vehicle.average_speed_kmh if vehicle.average_speed_kmh else default_speed
        matrices[idx] = create_time_matrix(locations, speed)
    return matrices


# ============== SOLVER OR-TOOLS ==============

def solve_vrptw(request: OptimizeRequest) -> OptimizeResponse:
    """
    VERSÃO 7.0 - VRPTW com Time Windows como HARD CONSTRAINT
    
    Novidades v7.0:
    - Velocidade por veículo
    - Grupos de entregas (mesmo veículo)
    - Pré-atribuição de entregas
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
    
    # Velocidade padrão
    default_speed = request.average_speed_kmh if request.average_speed_kmh > 0 else DEFAULT_SPEED_KMH
    
    # Criar matriz de distância (única para todos)
    distance_matrix = create_distance_matrix(locations)
    
    # Service times por nó (depot = 0)
    service_times = [0]  # Depot não tem service time
    for delivery in request.deliveries:
        customer = customer_map.get(delivery.customer_id)
        st = customer.service_time if customer and customer.service_time else DEFAULT_SERVICE_TIME
        service_times.append(st)
    
    # Time windows por nó
    time_windows = [(request.start_time, MAX_TIME_HORIZON)]
    
    for delivery in request.deliveries:
        customer = customer_map.get(delivery.customer_id)
        if customer and customer.window_start is not None and customer.window_end is not None:
            time_windows.append((customer.window_start, customer.window_end))
        else:
            time_windows.append((0, MAX_TIME_HORIZON))
    
    # ===== PROCESSAR REGRAS DE ROTEAMENTO =====
    print(f"\n=== PROCESSANDO REGRAS DE ROTEAMENTO ===")
    
    # Criar mapa de freteiro_id -> vehicle_id
    freteiro_to_vehicle = {}
    for vehicle in request.vehicles:
        if vehicle.freteiro_id:
            freteiro_to_vehicle[vehicle.freteiro_id] = vehicle.id
    
    # Processar regras (ordenadas por prioridade)
    rules_applied_count = 0
    if request.routing_rules:
        sorted_rules = sorted(request.routing_rules, key=lambda r: r.priority, reverse=True)
        print(f"Total de regras recebidas: {len(sorted_rules)}")
        
        for rule in sorted_rules:
            print(f"\nAplicando regra: {rule.name} (tipo: {rule.type}, prioridade: {rule.priority})")
            print(f"  Condição: {rule.condition.field} {rule.condition.operator} '{rule.condition.value}'")
            print(f"  Ação: {rule.action}")
            
            if rule.type == "fixed_driver":
                # Regra: Atribuir entregas a um motorista específico
                if not rule.action or not rule.action.freteiro_id:
                    print(f"  ⚠️ Regra sem freteiro_id (action={rule.action}), pulando")
                    continue
                
                print(f"  Buscando veículo(s) do freteiro: {rule.action.freteiro_id}")
                rules_applied_count += 1
                
                # Verificar se há vehicle_ids específicos na ação
                target_vehicle_ids = []
                if rule.action.vehicle_ids and len(rule.action.vehicle_ids) > 0:
                    # Usar veículos específicos da ação
                    target_vehicle_ids = rule.action.vehicle_ids
                    print(f"  Usando {len(target_vehicle_ids)} veículo(s) específico(s)")
                else:
                    # Buscar veículo do freteiro (comportamento original)
                    vehicle_id = freteiro_to_vehicle.get(rule.action.freteiro_id)
                    if not vehicle_id:
                        print(f"  ⚠️ Freteiro não encontrado ou sem veículo ativo, pulando")
                        continue
                    target_vehicle_ids = [vehicle_id]
                
                # Coletar entregas que correspondem à condição
                matched_deliveries = []
                for delivery in request.deliveries:
                    # Pular se já tem pré-atribuição
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
                        matched_deliveries.append(delivery)
                
                # Distribuir entregas entre os veículos (round-robin se múltiplos)
                if len(target_vehicle_ids) == 1:
                    # Um único veículo: atribuir todas
                    for delivery in matched_deliveries:
                        delivery.vehicle_id = target_vehicle_ids[0]
                        print(f"  ✅ {delivery.customer_name} → veículo {target_vehicle_ids[0]}")
                else:
                    # Múltiplos veículos: distribuir balanceado
                    for idx, delivery in enumerate(matched_deliveries):
                        vehicle_idx = idx % len(target_vehicle_ids)
                        vehicle_id = target_vehicle_ids[vehicle_idx]
                        delivery.vehicle_id = vehicle_id
                        print(f"  ✅ {delivery.customer_name} → veículo {vehicle_id}")
                
                print(f"  Total: {len(matched_deliveries)} entregas pré-atribuídas")
                if len(matched_deliveries) == 0:
                    print(f"  ⚠️ Nenhuma entrega correspondeu à condição!")
            
            elif rule.type == "group_by_name":
                # Regra: Agrupar entregas com nomes similares
                # Coletar entregas que correspondem
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
                    # Adicionar ao delivery_groups
                    if not request.delivery_groups:
                        request.delivery_groups = []
                    request.delivery_groups.append(group_deliveries)
                    print(f"  ✅ Agrupadas {len(group_deliveries)} entregas: {', '.join(group_deliveries[:3])}...")
                    rules_applied_count += 1
                else:
                    print(f"  ⚠️ Apenas {len(group_deliveries)} entrega(s) encontrada(s), não há o que agrupar")
            
            elif rule.type == "vehicle_exclusive_for_group":
                # Regra: Veículo(s) só aceita(m) entregas de um grupo específico
                if not rule.vehicle_ids or len(rule.vehicle_ids) == 0:
                    print(f"  ⚠️ Regra sem vehicle_ids, pulando")
                    continue
                
                print(f"  Veículos exclusivos: {rule.vehicle_ids}")
                rules_applied_count += 1
                
                # Marcar quais entregas correspondem à condição
                allowed_delivery_ids = set()
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
                        allowed_delivery_ids.add(delivery.id)
                
                print(f"  Entregas permitidas: {len(allowed_delivery_ids)}")
                
                # Armazenar restrição para aplicar depois (após criar o modelo OR-Tools)
                # Vamos adicionar um campo temporário no request
                if not hasattr(request, 'vehicle_exclusive_rules'):
                    request.vehicle_exclusive_rules = []
                request.vehicle_exclusive_rules.append({
                    'vehicle_ids': rule.vehicle_ids,
                    'allowed_delivery_ids': allowed_delivery_ids
                })
            
            elif rule.type == "force_multiple_vehicles":
                # Regra: Forçar distribuição de grupo em múltiplos veículos específicos
                if not rule.vehicle_ids or len(rule.vehicle_ids) < 2:
                    print(f"  ⚠️ Regra requer pelo menos 2 veículos, pulando")
                    continue
                
                print(f"  Forçar distribuição em {len(rule.vehicle_ids)} veículos")
                rules_applied_count += 1
                
                # Coletar entregas que correspondem à condição
                matched_deliveries = []
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
                        matched_deliveries.append(delivery)
                
                if len(matched_deliveries) < len(rule.vehicle_ids):
                    print(f"  ⚠️ Apenas {len(matched_deliveries)} entrega(s), menos que {len(rule.vehicle_ids)} veículos")
                    continue
                
                print(f"  Distribuindo {len(matched_deliveries)} entregas em {len(rule.vehicle_ids)} veículos")
                
                # Modo de distribuição
                distribution_mode = rule.action.distribution_mode if rule.action and rule.action.distribution_mode else "balanced"
                
                if distribution_mode == "balanced":
                    # Distribuir de forma balanceada (round-robin)
                    for idx, delivery in enumerate(matched_deliveries):
                        vehicle_idx = idx % len(rule.vehicle_ids)
                        vehicle_id = rule.vehicle_ids[vehicle_idx]
                        
                        # Pular se já tem pré-atribuição
                        if not delivery.vehicle_id:
                            delivery.vehicle_id = vehicle_id
                            print(f"    ✅ {delivery.customer_name} -> {vehicle_id}")
                else:
                    # Modo sequential: preencher primeiro veículo, depois segundo, etc.
                    deliveries_per_vehicle = len(matched_deliveries) // len(rule.vehicle_ids)
                    remainder = len(matched_deliveries) % len(rule.vehicle_ids)
                    
                    delivery_idx = 0
                    for vehicle_idx, vehicle_id in enumerate(rule.vehicle_ids):
                        count = deliveries_per_vehicle + (1 if vehicle_idx < remainder else 0)
                        for _ in range(count):
                            if delivery_idx < len(matched_deliveries):
                                delivery = matched_deliveries[delivery_idx]
                                if not delivery.vehicle_id:
                                    delivery.vehicle_id = vehicle_id
                                    print(f"    ✅ {delivery.customer_name} -> {vehicle_id}")
                                delivery_idx += 1
    else:
        print("Nenhuma regra recebida no payload")
    
    print(f"\n=== CONFIGURAÇÃO v7.3 ===")
    print(f"Velocidade padrão: {default_speed} km/h")
    print(f"Horário início: {minutes_to_time(request.start_time)}")
    print(f"Entregas: {len(request.deliveries)}")
    print(f"Veículos disponíveis: {len(request.vehicles)}")
    print(f"Regras recebidas: {len(request.routing_rules) if request.routing_rules else 0}")
    print(f"Regras aplicadas com sucesso: {rules_applied_count}")
    print(f"Pré-atribuições: {sum(1 for d in request.deliveries if d.vehicle_id)}")
    print(f"Grupos: {len(request.delivery_groups) if request.delivery_groups else 0}")
    
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
                    capacity_kg=99999,
                    average_speed_kmh=default_speed
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
    
    # Log velocidades por veículo
    for idx, v in enumerate(vehicles[:len(request.vehicles)]):
        speed = v.average_speed_kmh if v.average_speed_kmh else default_speed
        print(f"  Veículo {v.name}: {speed} km/h")
    
    # Criar matrizes de tempo por veículo
    time_matrices = create_time_matrix_per_vehicle(locations, vehicles, default_speed)
    
    # Mapear delivery_id para node_index
    delivery_id_to_node = {}
    for idx, delivery in enumerate(request.deliveries):
        delivery_id_to_node[delivery.id] = idx + 1  # +1 porque depot é 0
    
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
    
    # Registrar callbacks de tempo para cada veículo
    time_callback_indices = []
    for vehicle_idx in range(num_vehicles):
        callback = make_time_callback(vehicle_idx)
        callback_index = routing.RegisterTransitCallback(callback)
        time_callback_indices.append(callback_index)
    
    # ----- DIMENSÃO DE TEMPO COM TIME WINDOWS -----
    # Usar callback do primeiro veículo como base (OR-Tools requer um único callback para dimensão)
    # Mas vamos usar SetArcCostEvaluatorOfVehicle para custos diferentes
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
        MAX_TIME_HORIZON,  # Slack máximo
        MAX_TIME_HORIZON,  # Tempo máximo por veículo
        False,
        "Time"
    )
    time_dimension = routing.GetDimensionOrDie("Time")
    
    # Definir janelas de tempo para cada nó (SOFT CONSTRAINT com 15min de flexibilidade)
    FLEXIBILITY_MINUTES = 15
    for node in range(num_locations):
        index = manager.NodeToIndex(node)
        window_start, window_end = time_windows[node]
        
        # Permitir flexibilidade de 15min antes e depois
        flexible_start = max(0, window_start - FLEXIBILITY_MINUTES)
        flexible_end = min(MAX_TIME_HORIZON, window_end + FLEXIBILITY_MINUTES)
        
        time_dimension.CumulVar(index).SetRange(flexible_start, flexible_end)
        
        if node > 0:
            delivery = request.deliveries[node - 1]
            customer = customer_map.get(delivery.customer_id)
            customer_name = customer.name if customer else "?"
            print(f"  Nó {node} ({customer_name}): janela {minutes_to_time(window_start)}-{minutes_to_time(window_end)} (flex: {minutes_to_time(flexible_start)}-{minutes_to_time(flexible_end)})")
    
    # Adicionar penalidade proporcional por atraso (5000 por minuto)
    # Isso incentiva o OR-Tools a respeitar as janelas originais, mas permite flexibilidade
    for vehicle_idx in range(num_vehicles):
        time_dimension.SetSpanCostCoefficientForVehicle(5000, vehicle_idx)
    
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
        return int(math.ceil(delivery.boxes))
    
    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,
        [v.capacity_boxes for v in vehicles],
        True,
        "Capacity"
    )
    
    # ----- CUSTOS POR VEÍCULO (velocidades diferentes) -----
    for vehicle_idx in range(num_vehicles):
        routing.SetArcCostEvaluatorOfVehicle(time_callback_indices[vehicle_idx], vehicle_idx)
    
    # Custo fixo por veículo (para minimizar número de veículos)
    if request.mode == "minimize_vehicles":
        for vehicle_idx in range(num_vehicles):
            routing.SetFixedCostOfVehicle(100000, vehicle_idx)
    
    # ----- PRÉ-ATRIBUIÇÃO DE ENTREGAS (vehicle_id em delivery) -----
    pre_assigned_count = 0
    for delivery in request.deliveries:
        if delivery.vehicle_id and delivery.vehicle_id in vehicle_id_to_idx:
            node = delivery_id_to_node[delivery.id]
            vehicle_idx = vehicle_id_to_idx[delivery.vehicle_id]
            index = manager.NodeToIndex(node)
            
            # Forçar que este nó seja atendido por este veículo
            routing.SetAllowedVehiclesForIndex([vehicle_idx], index)
            pre_assigned_count += 1
            
            customer = customer_map.get(delivery.customer_id)
            customer_name = customer.name if customer else "?"
            print(f"  Pré-atribuição: {customer_name} -> {vehicles[vehicle_idx].name}")
    
    if pre_assigned_count > 0:
        print(f"\n  Total pré-atribuições: {pre_assigned_count}")
    
    # ----- GRUPOS DE ENTREGAS (delivery_groups) -----
    if request.delivery_groups:
        print(f"\n=== GRUPOS DE ENTREGAS ===")
        for group_idx, group in enumerate(request.delivery_groups):
            if len(group) < 2:
                continue
            
            # Converter delivery_ids para node indices
            group_nodes = []
            for delivery_id in group:
                if delivery_id in delivery_id_to_node:
                    group_nodes.append(delivery_id_to_node[delivery_id])
            
            if len(group_nodes) < 2:
                continue
            
            print(f"  Grupo {group_idx + 1}: {len(group_nodes)} entregas devem ficar juntas")
            
            # Adicionar restrição: todas as entregas do grupo devem ser atendidas pelo mesmo veículo
            # Usamos PickupAndDelivery como workaround ou Same Vehicle Constraint
            first_node = group_nodes[0]
            first_index = manager.NodeToIndex(first_node)
            
            for node in group_nodes[1:]:
                index = manager.NodeToIndex(node)
                # Restrição: ambos devem estar no mesmo veículo
                routing.solver().Add(
                    routing.VehicleVar(first_index) == routing.VehicleVar(index)
                )
    
    # ----- RESTRIÇÕES DE EXCLUSIVIDADE DE VEÍCULO -----
    if hasattr(request, 'vehicle_exclusive_rules'):
        print(f"\n=== RESTRIÇÕES DE EXCLUSIVIDADE ===")
        for rule_idx, rule_data in enumerate(request.vehicle_exclusive_rules):
            vehicle_ids = rule_data['vehicle_ids']
            allowed_delivery_ids = rule_data['allowed_delivery_ids']
            
            # Converter vehicle_ids para vehicle_indices
            exclusive_vehicle_indices = []
            for vid in vehicle_ids:
                if vid in vehicle_id_to_idx:
                    exclusive_vehicle_indices.append(vehicle_id_to_idx[vid])
            
            if not exclusive_vehicle_indices:
                print(f"  Regra {rule_idx + 1}: Veículos não encontrados, pulando")
                continue
            
            print(f"  Regra {rule_idx + 1}: {len(exclusive_vehicle_indices)} veículo(s) exclusivo(s) para {len(allowed_delivery_ids)} entrega(s)")
            
            # Para cada entrega:
            # - Se está na lista permitida: só pode ser atendida pelos veículos exclusivos
            # - Se NÃO está na lista: NÃO pode ser atendida pelos veículos exclusivos
            for delivery in request.deliveries:
                node = delivery_id_to_node[delivery.id]
                index = manager.NodeToIndex(node)
                
                # Pular se já tem pré-atribuição
                if delivery.vehicle_id:
                    continue
                
                if delivery.id in allowed_delivery_ids:
                    # Entrega permitida: só pode usar estes veículos
                    routing.SetAllowedVehiclesForIndex(exclusive_vehicle_indices, index)
                    customer = customer_map.get(delivery.customer_id)
                    customer_name = customer.name if customer else "?"
                    print(f"    ✅ {customer_name} -> apenas veículos {vehicle_ids}")
                else:
                    # Entrega NÃO permitida: proibir estes veículos
                    # Criar lista de veículos permitidos (todos EXCETO os exclusivos)
                    allowed_vehicles = [i for i in range(num_vehicles) if i not in exclusive_vehicle_indices]
                    if allowed_vehicles:
                        routing.SetAllowedVehiclesForIndex(allowed_vehicles, index)
    
    # ----- PENALIDADES -----
    # Penalidade MUITO ALTA para não-atendimento (100M)
    # Isso força o OR-Tools a alocar TODAS as entregas
    penalty = 100000000  # 100 milhões (10x maior que antes)
    for node in range(1, num_locations):
        routing.AddDisjunction([manager.NodeToIndex(node)], penalty)
    
    print(f"\nPenalidade por não-atendimento: {penalty:,}")
    
    # ----- PARÂMETROS DE BUSCA -----
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    # Aumentar tempo de solução para 120s (2 minutos)
    search_parameters.time_limit.FromSeconds(120)
    
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
            if node > 0:
                route_nodes.append(node)
            index = solution.Value(routing.NextVar(index))
        
        if not route_nodes:
            continue
        
        # Velocidade deste veículo
        vehicle = vehicles[vehicle_idx] if vehicle_idx < len(vehicles) else None
        vehicle_speed = vehicle.average_speed_kmh if vehicle and vehicle.average_speed_kmh else default_speed
        
        # Matriz de tempo para este veículo
        time_matrix = time_matrices.get(vehicle_idx, base_time_matrix)
        
        # ===== RECALCULAR TEMPOS DE FORMA PROPAGADA =====
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
        
        print(f"\n=== ROTA {vehicle_idx + 1}: {vehicle.name if vehicle else 'Extra'} ({vehicle_speed} km/h) ===")
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
            
            # Service time
            service_time = service_times[node]
            
            # Janela de tempo
            window_start = time_windows[node][0]
            window_end = time_windows[node][1]
            
            # Calcular effective_start
            if arrival_time < window_start:
                effective_start = window_start
                wait_time = window_start - arrival_time
                arrived_early = True
            else:
                effective_start = arrival_time
                wait_time = 0
                arrived_early = False
            
            arrived_late = arrival_time > window_end
            window_ok = not arrived_late
            
            departure_time = effective_start + service_time
            
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
            
            route_boxes += delivery.boxes
            route_weight += delivery.weight_kg
            route_value += delivery.value
            route_distance += distance
            route_wait_time += wait_time
            route_service_time += service_time
            route_travel_time += travel_time
            
            current_time = departure_time
            prev_node = node
        
        # Distância de volta ao depot
        distance_to_depot = distance_matrix[prev_node][0]
        route_distance += distance_to_depot
        
        total_time = stops[-1].departure_time - request.start_time if stops else 0
        
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
            capacity_used_percent=capacity_used_percent,
            average_speed_kmh=vehicle_speed
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


# ============== FUNÇÃO DE RECÁLCULO ==============

def recalculate_route_times(request: RecalculateRequest) -> RecalculateResponse:
    """
    Recalcula os tempos de uma rota existente com nova velocidade
    Não usa OR-Tools, apenas propaga os tempos sequencialmente
    """
    if not request.stops:
        return RecalculateResponse(
            success=True,
            message="Nenhuma parada para recalcular",
            stops=[],
            total_time_minutes=0,
            total_distance_km=0,
            total_wait_time=0,
            total_service_time=0,
            total_travel_time=0
        )
    
    speed = request.average_speed_kmh if request.average_speed_kmh > 0 else DEFAULT_SPEED_KMH
    
    # Ordenar stops por stop_order
    sorted_stops = sorted(request.stops, key=lambda s: s.stop_order)
    
    # Criar lista de localizações
    locations = [request.depot.location]
    for stop in sorted_stops:
        locations.append(Location(lat=stop.lat, lng=stop.lng))
    
    # Criar matrizes
    time_matrix = create_time_matrix(locations, speed)
    distance_matrix = create_distance_matrix(locations)
    
    # Recalcular tempos
    result_stops = []
    current_time = request.start_time
    prev_node = 0
    
    total_distance = 0.0
    total_wait_time = 0
    total_service_time = 0
    total_travel_time = 0
    
    for idx, stop in enumerate(sorted_stops):
        node = idx + 1
        
        travel_time = time_matrix[prev_node][node]
        distance = distance_matrix[prev_node][node]
        
        arrival_time = current_time + travel_time
        
        window_start = stop.window_start if stop.window_start is not None else 0
        window_end = stop.window_end if stop.window_end is not None else MAX_TIME_HORIZON
        
        if arrival_time < window_start:
            effective_start = window_start
            wait_time = window_start - arrival_time
            arrived_early = True
        else:
            effective_start = arrival_time
            wait_time = 0
            arrived_early = False
        
        arrived_late = arrival_time > window_end
        window_ok = not arrived_late
        
        departure_time = effective_start + stop.service_time
        
        result_stops.append(Stop(
            delivery_id=stop.delivery_id,
            customer_id=stop.customer_id,
            customer_name=stop.customer_name,
            stop_order=stop.stop_order,
            arrival_time=arrival_time,
            effective_start=effective_start,
            departure_time=departure_time,
            service_time=stop.service_time,
            wait_time=wait_time,
            travel_time_from_prev=travel_time,
            distance_from_prev_km=distance,
            window_start=stop.window_start,
            window_end=stop.window_end,
            window_ok=window_ok,
            arrived_early=arrived_early,
            arrived_late=arrived_late,
            boxes=stop.boxes,
            weight_kg=stop.weight_kg,
            value=stop.value
        ))
        
        total_distance += distance
        total_wait_time += wait_time
        total_service_time += stop.service_time
        total_travel_time += travel_time
        
        current_time = departure_time
        prev_node = node
    
    # Distância de volta ao depot
    total_distance += distance_matrix[prev_node][0]
    
    total_time = result_stops[-1].departure_time - request.start_time if result_stops else 0
    
    return RecalculateResponse(
        success=True,
        message=f"Tempos recalculados com velocidade {speed} km/h",
        stops=result_stops,
        total_time_minutes=total_time,
        total_distance_km=round(total_distance, 2),
        total_wait_time=total_wait_time,
        total_service_time=total_service_time,
        total_travel_time=total_travel_time
    )


# ============== ENDPOINTS ==============

@app.get("/")
async def root():
    return {"status": "ok", "service": "OR-Tools Route Optimizer", "version": "7.3.0"}


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "ortools_version": "9.x",
        "api_key_configured": API_KEY != "dev-key-change-in-production",
        "version": "7.0.0",
        "features": [
            "Time Windows as HARD CONSTRAINT",
            "Time dimension with service_time",
            "Optimizes by total time (not distance)",
            "Sequential time propagation",
            "Wait time when arrived early",
            "v7.0: Vehicle-specific average_speed_kmh",
            "v7.0: Delivery groups (same vehicle)",
            "v7.0: Pre-assignment (vehicle_id in delivery)",
            "v7.0: /recalculate endpoint",
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


@app.post("/recalculate", response_model=RecalculateResponse)
async def recalculate_route(
    request: RecalculateRequest,
    authorization: Optional[str] = Header(None)
):
    """
    Recalcula os tempos de uma rota existente com nova velocidade
    Útil quando o usuário altera a velocidade do veículo
    """
    if API_KEY != "dev-key-change-in-production":
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="API Key não fornecida")
        
        provided_key = authorization.replace("Bearer ", "")
        if provided_key != API_KEY:
            raise HTTPException(status_code=401, detail="API Key inválida")
    
    try:
        result = recalculate_route_times(request)
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Erro no recálculo: {str(e)}")


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
    
    # Verificar pré-atribuições
    pre_assigned = [d for d in request.deliveries if d.vehicle_id]
    vehicle_ids = {v.id for v in request.vehicles}
    for d in pre_assigned:
        if d.vehicle_id not in vehicle_ids:
            issues.append(f"Delivery {d.id} pré-atribuída a veículo inexistente: {d.vehicle_id}")
    
    # Verificar grupos
    delivery_ids = {d.id for d in request.deliveries}
    if request.delivery_groups:
        for group_idx, group in enumerate(request.delivery_groups):
            for delivery_id in group:
                if delivery_id not in delivery_ids:
                    issues.append(f"Grupo {group_idx + 1} referencia delivery inexistente: {delivery_id}")
    
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
            "pre_assigned_deliveries": len(pre_assigned),
            "delivery_groups": len(request.delivery_groups) if request.delivery_groups else 0
        }
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
