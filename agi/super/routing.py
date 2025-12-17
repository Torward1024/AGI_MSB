from typing import Dict, Any, Optional, List, Tuple, Union, Set
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime, timedelta
from collections import defaultdict, deque
import threading
import asyncio
from concurrent.futures import ThreadPoolExecutor
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import heapq
import random

from common.utils.logging_setup import logger
from common.super.super import Super
from agi.nodeentity import NodeEntity
from agi.edgeentity import EdgeEntity
from agi.knowledgechunk import KnowledgeChunk
from agi.clustercontainer import ClusterContainer

class RoutingSuper(Super):
    """
    Интеллектуальная маршрутизация в графовой архитектуре AGI.
    
    Реализует механизмы поиска оптимальных путей:
    - GNN-based предсказание путей
    - Monte Carlo Tree Search
    - Real-time адаптивная перемаршрутизация
    """
    
    OPERATION = "routing"
    
    def __init__(self, manipulator=None, methods=None, cache_size=2048):
        super().__init__(manipulator, methods, cache_size)
        self._path_cache = {}
        self._routing_history = deque(maxlen=100)
        self._failure_patterns = {}
    
    def _routing_cluster(self, cluster: ClusterContainer, 
                        attributes: Dict[str, Any]) -> Dict[str, Any]:
        """Маршрутизация в кластере с выбором стратегии"""
        routing_operation = attributes.get("operation", "predict_path")
        
        if routing_operation == "predict_path":
            return self._routing_predict_optimal_path(cluster, attributes)
        elif routing_operation == "adaptive_rerouting":
            return self._routing_adaptive_rerouting(cluster, attributes)
        elif routing_operation == "meta_learning":
            return self._routing_meta_learning(cluster, attributes)
        else:
            return self._build_response(cluster, False, None, None, 
                                      f"Unknown routing operation: {routing_operation}")
    
    def _routing_predict_optimal_path(self, cluster: ClusterContainer, 
                                    attributes: Dict[str, Any]) -> Dict[str, Any]:
        """
        Предсказание оптимального пути через GNN и MCTS.
        
        Использует GNN для предсказания путей,
        Monte Carlo Tree Search для оптимизации,
        cost-benefit analysis (latency vs accuracy).
        """
        if not cluster.isactive:
            return self._build_response(cluster, False, None, None, "Cluster is inactive")
        
        start_node = attributes.get("start_node")
        target_node = attributes.get("target_node")
        query_vector = attributes.get("query_vector")
        max_depth = attributes.get("max_depth", 5)
        search_iterations = attributes.get("search_iterations", 100)
        
        if not start_node or not target_node:
            return self._build_response(cluster, False, None, None, 
                                      "Missing start_node or target_node")
        
        start_time = datetime.now()
        
        # Проверяем кэш
        cache_key = f"{cluster.name}:{start_node}:{target_node}:{hash(str(query_vector)) if query_vector else 'none'}"
        if cache_key in self._path_cache:
            cached_result = self._path_cache[cache_key]
            cached_result["from_cache"] = True
            return self._build_response(cluster, True, "predict_path", cached_result)
        
        # Шаг 1: GNN-based предсказание путей (упрощенное)
        gnn_predictions = self._gnn_path_prediction(
            cluster, start_node, target_node, query_vector, max_depth
        )
        
        # Шаг 2: Monte Carlo Tree Search для оптимизации
        mcts_paths = self._monte_carlo_tree_search(
            cluster, start_node, target_node, max_depth, search_iterations
        )
        
        # Шаг 3: Cost-Benefit Analysis
        evaluated_paths = self._evaluate_paths_cost_benefit(
            cluster, gnn_predictions + mcts_paths
        )
        
        # Шаг 4: Выбор оптимального пути
        optimal_path = self._select_optimal_path(evaluated_paths)
        
        # Шаг 5: Подготовка пути к исполнению
        execution_plan = self._prepare_execution_plan(cluster, optimal_path)
        
        latency = (datetime.now() - start_time).total_seconds()
        
        # Кэшируем результат
        result = {
            "status": "success",
            "cluster": cluster.name,
            "start_node": start_node,
            "target_node": target_node,
            "gnn_predictions": len(gnn_predictions),
            "mcts_paths_evaluated": len(mcts_paths),
            "optimal_path": optimal_path["path"],
            "path_cost": optimal_path["cost"],
            "path_benefit": optimal_path["benefit"],
            "confidence": optimal_path["confidence"],
            "execution_plan": execution_plan,
            "search_time_seconds": latency
        }
        
        self._path_cache[cache_key] = result
        self._routing_history.append({
            "timestamp": datetime.now(),
            "cluster": cluster.name,
            "start": start_node,
            "target": target_node,
            "path_length": len(optimal_path["path"]),
            "cost": optimal_path["cost"],
            "confidence": optimal_path["confidence"]
        })
        
        logger.info(f"Path prediction completed for '{cluster.name}': "
                   f"{start_node} -> {target_node}, path length={len(optimal_path['path'])}, "
                   f"confidence={optimal_path['confidence']:.3f}")
        
        return self._build_response(cluster, True, "predict_path", result)
    
    def _routing_adaptive_rerouting(self, cluster: ClusterContainer, 
                                  attributes: Dict[str, Any]) -> Dict[str, Any]:
        """
        Real-time адаптивная перемаршрутизация.
        
        Оптимизация путей в реальном времени,
        обнаружение и восстановление при сбоях,
        балансировка нагрузки по параллельным путям.
        """
        if not cluster.isactive:
            return self._build_response(cluster, False, None, None, "Cluster is inactive")
        
        current_path = attributes.get("current_path", [])
        failed_nodes = attributes.get("failed_nodes", [])
        load_info = attributes.get("load_info", {})
        optimization_mode = attributes.get("optimization_mode", "recovery")
        
        start_time = datetime.now()
        
        if not current_path or len(current_path) < 2:
            return self._build_response(cluster, False, None, None, 
                                      "Invalid current path provided")
        
        # Шаг 1: Анализ текущего пути
        path_analysis = self._analyze_current_path(
            cluster, current_path, failed_nodes, load_info
        )
        
        # Шаг 2: Обнаружение сбоев и bottlenecks
        issues_detected = self._detect_routing_issues(path_analysis)
        
        # Шаг 3: Генерация альтернативных путей
        alternative_paths = []
        
        if optimization_mode == "recovery" and issues_detected["failures"]:
            # Режим восстановления: обход failed nodes
            alternative_paths = self._generate_recovery_paths(
                cluster, current_path, failed_nodes
            )
        elif optimization_mode == "optimization" and issues_detected["bottlenecks"]:
            # Режим оптимизации: обход bottlenecks
            alternative_paths = self._generate_optimization_paths(
                cluster, current_path, path_analysis["bottlenecks"]
            )
        elif optimization_mode == "load_balancing":
            # Режим балансировки нагрузки
            alternative_paths = self._generate_load_balanced_paths(
                cluster, current_path, load_info
            )
        
        # Шаг 4: Оценка альтернативных путей
        evaluated_alternatives = self._evaluate_alternative_paths(
            cluster, alternative_paths, path_analysis
        )
        
        # Шаг 5: Выбор лучшего альтернативного пути
        best_alternative = None
        if evaluated_alternatives:
            best_alternative = max(
                evaluated_alternatives, 
                key=lambda p: p["improvement_score"]
            )
        
        # Шаг 6: Применение перемаршрутизации
        rerouting_result = self._apply_rerouting(
            cluster, current_path, best_alternative, issues_detected
        )
        
        latency = (datetime.now() - start_time).total_seconds()
        
        result = {
            "status": "success",
            "cluster": cluster.name,
            "optimization_mode": optimization_mode,
            "original_path": current_path,
            "path_analysis": path_analysis,
            "issues_detected": issues_detected,
            "alternatives_generated": len(alternative_paths),
            "best_alternative": best_alternative["path"] if best_alternative else None,
            "improvement_expected": best_alternative["improvement_score"] if best_alternative else 0,
            "rerouting_applied": rerouting_result["applied"],
            "rerouting_result": rerouting_result,
            "processing_time_seconds": latency
        }
        
        # Обновляем статистику сбоев
        if issues_detected["failures"]:
            for node in issues_detected["failures"]:
                if node not in self._failure_patterns:
                    self._failure_patterns[node] = {"count": 0, "last_failure": None}
                self._failure_patterns[node]["count"] += 1
                self._failure_patterns[node]["last_failure"] = datetime.now()
        
        logger.info(f"Adaptive rerouting completed for '{cluster.name}': "
                   f"mode={optimization_mode}, issues={len(issues_detected['failures'])} failures, "
                   f"{len(issues_detected['bottlenecks'])} bottlenecks, "
                   f"improvement={best_alternative['improvement_score'] if best_alternative else 0:.3f}")
        
        return self._build_response(cluster, True, "adaptive_rerouting", result)
    
    def _routing_meta_learning(self, cluster: ClusterContainer, 
                             attributes: Dict[str, Any]) -> Dict[str, Any]:
        """
        Meta-learning паттернов маршрутизации.
        
        Обучение паттернов маршрутизации на различных задачах,
        обобщение на новые типы запросов,
        эволюционная оптимизация routing heuristics.
        """
        if not cluster.isactive:
            return self._build_response(cluster, False, None, None, "Cluster is inactive")
        
        learning_mode = attributes.get("learning_mode", "pattern_analysis")
        training_data = attributes.get("training_data", self._routing_history)
        generalization_test = attributes.get("generalization_test", {})
        
        start_time = datetime.now()
        
        # Шаг 1: Анализ истории маршрутизации
        pattern_analysis = self._analyze_routing_patterns(training_data)
        
        # Шаг 2: Извлечение routing heuristics
        learned_heuristics = self._extract_routing_heuristics(pattern_analysis)
        
        # Шаг 3: Обобщение на новые типы запросов
        generalization_results = {}
        if generalization_test:
            generalization_results = self._test_generalization(
                cluster, learned_heuristics, generalization_test
            )
        
        # Шаг 4: Эволюционная оптимизация heuristics
        optimized_heuristics = self._evolve_routing_heuristics(
            learned_heuristics, pattern_analysis
        )
        
        # Шаг 5: Обновление routing models
        update_result = self._update_routing_models(
            cluster, optimized_heuristics, pattern_analysis
        )
        
        latency = (datetime.now() - start_time).total_seconds()
        
        result = {
            "status": "success",
            "cluster": cluster.name,
            "learning_mode": learning_mode,
            "training_samples": len(training_data),
            "patterns_identified": pattern_analysis["patterns_found"],
            "learned_heuristics": len(learned_heuristics),
            "generalization_results": generalization_results,
            "optimized_heuristics": len(optimized_heuristics),
            "models_updated": update_result["models_updated"],
            "improvement_expected": update_result["expected_improvement"],
            "learning_time_seconds": latency
        }
        
        logger.info(f"Routing meta-learning completed for '{cluster.name}': "
                   f"{pattern_analysis['patterns_found']} patterns identified, "
                   f"{len(optimized_heuristics)} heuristics optimized")
        
        return self._build_response(cluster, True, "meta_learning", result)
    
    # Вспомогательные методы для _routing_predict_optimal_path
    def _gnn_path_prediction(self, cluster: ClusterContainer,
                           start_node: str, target_node: str,
                           query_vector: Optional[np.ndarray],
                           max_depth: int) -> List[List[str]]:
        """GNN-based предсказание путей (упрощенная реализация)"""
        predicted_paths = []
        
        # Базовый BFS для поиска путей
        all_paths = self._find_all_paths_bfs(cluster, start_node, target_node, max_depth)
        
        if not all_paths:
            return predicted_paths
        
        # Упрощенная "GNN" оценка: фильтрация на основе характеристик узлов
        for path in all_paths:
            if self._evaluate_path_with_simple_gnn(cluster, path, query_vector):
                predicted_paths.append(path)
        
        # Ограничиваем количество предсказаний
        return predicted_paths[:10]
    
    def _find_all_paths_bfs(self, cluster: ClusterContainer,
                          start: str, target: str, max_depth: int) -> List[List[str]]:
        """Поиск всех путей с помощью BFS"""
        if not cluster.has_item(start) or not cluster.has_item(target):
            return []
        
        paths = []
        queue = deque([([start], 0)])  # (path, depth)
        
        while queue:
            path, depth = queue.popleft()
            current_node = path[-1]
            
            if current_node == target:
                paths.append(path)
                continue
            
            if depth >= max_depth:
                continue
            
            # Находим соседей через edges
            for edge_name in cluster.intra_cluster_edges:
                if cluster.has_item(edge_name):
                    edge = cluster.get(edge_name)
                    if isinstance(edge, EdgeEntity) and edge.source_node == current_node:
                        next_node = edge.target_node
                        if next_node not in path:  # Избегаем циклов
                            new_path = path + [next_node]
                            queue.append((new_path, depth + 1))
        
        return paths
    
    def _evaluate_path_with_simple_gnn(self, cluster: ClusterContainer,
                                     path: List[str],
                                     query_vector: Optional[np.ndarray]) -> bool:
        """Упрощенная оценка пути (заглушка для GNN)"""
        if len(path) < 2:
            return False
        
        # Проверяем, что все узлы в пути активны
        for node_name in path:
            if cluster.has_item(node_name):
                node = cluster.get(node_name)
                if isinstance(node, NodeEntity) and not node.isactive:
                    return False
            else:
                return False
        
        # Проверяем, что есть связи между последовательными узлами
        for i in range(len(path) - 1):
            source, target = path[i], path[i+1]
            connection_exists = False
            
            for edge_name in cluster.intra_cluster_edges:
                if cluster.has_item(edge_name):
                    edge = cluster.get(edge_name)
                    if isinstance(edge, EdgeEntity):
                        if (edge.source_node == source and edge.target_node == target) or \
                           (edge.source_node == target and edge.target_node == source):
                            connection_exists = True
                            break
            
            if not connection_exists:
                return False
        
        # Базовая проверка качества пути
        if len(path) > 10:  # Слишком длинный путь
            return False
        
        return True
    
    def _monte_carlo_tree_search(self, cluster: ClusterContainer,
                               start: str, target: str,
                               max_depth: int, iterations: int) -> List[List[str]]:
        """Monte Carlo Tree Search для поиска путей"""
        best_paths = []
        
        for _ in range(min(iterations, 50)):  # Ограничиваем итерации
            # Random walk поиск
            path = self._random_walk_search(cluster, start, target, max_depth)
            if path and path not in best_paths:
                best_paths.append(path)
        
        return best_paths[:5]  # Возвращаем только 5 лучших
    
    def _random_walk_search(self, cluster: ClusterContainer,
                          start: str, target: str, max_depth: int) -> List[str]:
        """Random walk поиск пути"""
        path = [start]
        visited = {start}
        current = start
        
        for _ in range(max_depth):
            if current == target:
                return path
            
            # Находим возможные следующие узлы
            next_nodes = []
            for edge_name in cluster.intra_cluster_edges:
                if cluster.has_item(edge_name):
                    edge = cluster.get(edge_name)
                    if isinstance(edge, EdgeEntity) and edge.source_node == current:
                        if edge.target_node not in visited:
                            next_nodes.append(edge.target_node)
            
            if not next_nodes:
                break
            
            # Выбираем случайный следующий узел
            next_node = random.choice(next_nodes)
            path.append(next_node)
            visited.add(next_node)
            current = next_node
        
        return [] if current != target else path
    
    def _evaluate_paths_cost_benefit(self, cluster: ClusterContainer,
                                   paths: List[List[str]]) -> List[Dict[str, Any]]:
        """Cost-benefit анализ путей"""
        evaluated_paths = []
        
        for path in paths:
            if len(path) < 2:
                continue
            
            # Вычисляем cost (латентность, сложность)
            cost = self._calculate_path_cost(cluster, path)
            
            # Вычисляем benefit (качество, уверенность)
            benefit = self._calculate_path_benefit(cluster, path)
            
            # Вычисляем confidence
            confidence = self._calculate_path_confidence(cluster, path)
            
            evaluated_paths.append({
                "path": path,
                "cost": cost,
                "benefit": benefit,
                "confidence": confidence,
                "score": benefit / max(0.001, cost)  # Простой score
            })
        
        return evaluated_paths
    
    def _calculate_path_cost(self, cluster: ClusterContainer, path: List[str]) -> float:
        """Расчет стоимости пути"""
        if len(path) < 2:
            return float('inf')
        
        total_cost = 0.0
        
        for i in range(len(path) - 1):
            source, target = path[i], path[i+1]
            
            # Базовая стоимость за переход
            transition_cost = 1.0
            
            # Учет задержки связи, если есть
            for edge_name in cluster.intra_cluster_edges:
                if cluster.has_item(edge_name):
                    edge = cluster.get(edge_name)
                    if isinstance(edge, EdgeEntity):
                        if (edge.source_node == source and edge.target_node == target) or \
                           (edge.source_node == target and edge.target_node == source):
                            transition_cost += edge.transmission_delay * 0.01
                            break
            
            # Учет энергии узлов
            if cluster.has_item(source):
                node = cluster.get(source)
                if isinstance(node, NodeEntity):
                    energy_factor = 1.0 / max(0.1, node.energy_level)
                    transition_cost *= energy_factor
            
            total_cost += transition_cost
        
        return total_cost
    
    def _calculate_path_benefit(self, cluster: ClusterContainer, path: List[str]) -> float:
        """Расчет пользы пути"""
        if not path:
            return 0.0
        
        total_benefit = 0.0
        
        for node_name in path:
            if cluster.has_item(node_name):
                node = cluster.get(node_name)
                if isinstance(node, NodeEntity):
                    # Польза основана на качестве узла
                    stats = node.get_stats()
                    node_benefit = (
                        stats.get("success_rate", 0.5) * 0.4 +
                        (node.energy_level * 0.3) +
                        min(1.0, node.activation_count / 100.0) * 0.2 +
                        0.1  # Базовая польза
                    )
                    total_benefit += node_benefit
        
        return total_benefit / max(1, len(path))  # Нормализуем
    
    def _calculate_path_confidence(self, cluster: ClusterContainer, path: List[str]) -> float:
        """Расчет уверенности в пути"""
        if len(path) < 2:
            return 0.0
        
        confidence_factors = []
        
        # Фактор 1: Надежность узлов
        node_reliability = []
        for node_name in path:
            if cluster.has_item(node_name):
                node = cluster.get(node_name)
                if isinstance(node, NodeEntity):
                    reliability = (
                        node.energy_level * 0.4 +
                        min(1.0, node.activation_count / 50.0) * 0.3 +
                        0.3  # Базовая надежность
                    )
                    node_reliability.append(reliability)
        
        if node_reliability:
            confidence_factors.append(np.mean(node_reliability) * 0.5)
        
        # Фактор 2: Качество связей
        connection_strength = []
        for i in range(len(path) - 1):
            source, target = path[i], path[i+1]
            max_strength = 0.0
            
            for edge_name in cluster.intra_cluster_edges:
                if cluster.has_item(edge_name):
                    edge = cluster.get(edge_name)
                    if isinstance(edge, EdgeEntity):
                        if (edge.source_node == source and edge.target_node == target) or \
                           (edge.source_node == target and edge.target_node == source):
                            strength = edge.get_connection_strength()
                            max_strength = max(max_strength, strength)
            
            connection_strength.append(max_strength)
        
        if connection_strength:
            confidence_factors.append(np.mean(connection_strength) * 0.3)
        
        # Фактор 3: Длина пути (более короткие пути лучше)
        length_factor = 1.0 / (1.0 + len(path) / 10.0)
        confidence_factors.append(length_factor * 0.2)
        
        return sum(confidence_factors)
    
    def _select_optimal_path(self, evaluated_paths: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Выбор оптимального пути"""
        if not evaluated_paths:
            return {"path": [], "cost": float('inf'), "benefit": 0, "confidence": 0, "score": 0}
        
        # Выбираем путь с наилучшим score
        best_path = max(evaluated_paths, key=lambda p: p["score"])
        
        return best_path
    
    def _prepare_execution_plan(self, cluster: ClusterContainer,
                              optimal_path: Dict[str, Any]) -> Dict[str, Any]:
        """Подготовка плана исполнения для оптимального пути"""
        path = optimal_path["path"]
        
        if len(path) < 2:
            return {"executable": False, "reason": "Path too short"}
        
        execution_steps = []
        total_estimated_time = 0.0
        total_estimated_energy = 0.0
        
        for i in range(len(path) - 1):
            source, target = path[i], path[i+1]
            
            step = {
                "step": i + 1,
                "from": source,
                "to": target,
                "action": "activate_and_transmit"
            }
            
            # Оценка времени и энергии
            time_estimate = 0.1  # Базовое время
            energy_estimate = 0.05  # Базовая энергия
            
            # Уточняем оценку на основе характеристик
            if cluster.has_item(source):
                node = cluster.get(source)
                if isinstance(node, NodeEntity):
                    energy_estimate += (1.0 - node.energy_level) * 0.02
            
            # Проверяем наличие связи
            for edge_name in cluster.intra_cluster_edges:
                if cluster.has_item(edge_name):
                    edge = cluster.get(edge_name)
                    if isinstance(edge, EdgeEntity):
                        if (edge.source_node == source and edge.target_node == target) or \
                           (edge.source_node == target and edge.target_node == source):
                            time_estimate += edge.transmission_delay * 0.001
                            break
            
            step["time_estimate"] = time_estimate
            step["energy_estimate"] = energy_estimate
            
            total_estimated_time += time_estimate
            total_estimated_energy += energy_estimate
            
            execution_steps.append(step)
        
        return {
            "executable": True,
            "path_length": len(path),
            "execution_steps": execution_steps,
            "total_estimated_time": total_estimated_time,
            "total_estimated_energy": total_estimated_energy,
            "confidence": optimal_path["confidence"]
        }
    
    # Вспомогательные методы для _routing_adaptive_rerouting
    def _analyze_current_path(self, cluster: ClusterContainer,
                            current_path: List[str],
                            failed_nodes: List[str],
                            load_info: Dict[str, float]) -> Dict[str, Any]:
        """Анализ текущего пути"""
        analysis = {
            "path_nodes": current_path,
            "path_length": len(current_path),
            "failed_nodes_in_path": [node for node in failed_nodes if node in current_path],
            "node_loads": {},
            "bottlenecks": [],
            "weak_links": [],
            "overall_health": 1.0
        }
        
        # Анализируем каждый узел в пути
        health_factors = []
        
        for i, node_name in enumerate(current_path):
            if cluster.has_item(node_name):
                node = cluster.get(node_name)
                if isinstance(node, NodeEntity):
                    # Проверяем статус узла
                    node_health = self._assess_node_health(node, node_name in failed_nodes)
                    health_factors.append(node_health)
                    
                    # Проверяем нагрузку
                    if node_name in load_info:
                        load = load_info[node_name]
                        analysis["node_loads"][node_name] = load
                        if load > 0.8:
                            analysis["bottlenecks"].append({
                                "node": node_name,
                                "load": load,
                                "position": i
                            })
                    
                    # Проверяем связи с предыдущим узлом (если не первый)
                    if i > 0:
                        prev_node = current_path[i-1]
                        link_health = self._assess_link_health(cluster, prev_node, node_name)
                        if link_health < 0.5:
                            analysis["weak_links"].append({
                                "from": prev_node,
                                "to": node_name,
                                "health": link_health
                            })
        
        # Вычисляем общее здоровье пути
        if health_factors:
            analysis["overall_health"] = np.mean(health_factors)
        
        # Определяем серьезность проблем
        analysis["severity"] = self._calculate_path_severity(analysis)
        
        return analysis
    
    def _assess_node_health(self, node: NodeEntity, is_failed: bool) -> float:
        """Оценка здоровья узла"""
        if is_failed or not node.isactive:
            return 0.0
        
        health_factors = []
        
        # Фактор 1: Энергия
        health_factors.append(node.energy_level * 0.4)
        
        # Фактор 2: Активность
        activity_factor = min(1.0, node.activation_count / 100.0)
        health_factors.append(activity_factor * 0.3)
        
        # Фактор 3: Успешность
        stats = node.get_stats()
        success_factor = stats.get("success_rate", 0.5)
        health_factors.append(success_factor * 0.3)
        
        return sum(health_factors)
    
    def _assess_link_health(self, cluster: ClusterContainer,
                          source: str, target: str) -> float:
        """Оценка здоровья связи между узлами"""
        best_health = 0.0
        
        for edge_name in cluster.intra_cluster_edges:
            if cluster.has_item(edge_name):
                edge = cluster.get(edge_name)
                if isinstance(edge, EdgeEntity):
                    if (edge.source_node == source and edge.target_node == target) or \
                       (edge.source_node == target and edge.target_node == source):
                        
                        health_factors = []
                        
                        # Фактор 1: Вес связи
                        health_factors.append(abs(edge.weight) * 0.3)
                        
                        # Фактор 2: Частота использования
                        usage_factor = min(1.0, edge.usage_frequency / 50.0)
                        health_factors.append(usage_factor * 0.3)
                        
                        # Фактор 3: Эволюционная оценка
                        health_factors.append(edge.evolutionary_score * 0.2)
                        
                        # Фактор 4: Недавняя активность
                        time_since_last = (datetime.now() - edge.last_reinforcement_time).total_seconds()
                        recency_factor = 1.0 / (1.0 + time_since_last / 3600)
                        health_factors.append(recency_factor * 0.2)
                        
                        link_health = sum(health_factors)
                        best_health = max(best_health, link_health)
        
        return best_health
    
    def _calculate_path_severity(self, path_analysis: Dict[str, Any]) -> float:
        """Расчет серьезности проблем на пути"""
        severity = 0.0
        
        # Failed nodes - самый серьезный
        severity += len(path_analysis["failed_nodes_in_path"]) * 0.4
        
        # Bottlenecks
        for bottleneck in path_analysis["bottlenecks"]:
            severity += (bottleneck["load"] - 0.8) * 0.5  # Чем выше нагрузка, тем серьезнее
        
        # Weak links
        severity += len(path_analysis["weak_links"]) * 0.2
        
        # Общее здоровье
        severity += (1.0 - path_analysis["overall_health"]) * 0.3
        
        return min(1.0, severity)
    
    def _detect_routing_issues(self, path_analysis: Dict[str, Any]) -> Dict[str, List]:
        """Обнаружение проблем маршрутизации"""
        issues = {
            "failures": path_analysis["failed_nodes_in_path"],
            "bottlenecks": [b["node"] for b in path_analysis["bottlenecks"]],
            "weak_links": [f"{link['from']}->{link['to']}" for link in path_analysis["weak_links"]],
            "overall_severity": path_analysis["severity"]
        }
        
        # Классифицируем серьезность
        if path_analysis["severity"] > 0.7:
            issues["severity_level"] = "critical"
        elif path_analysis["severity"] > 0.4:
            issues["severity_level"] = "high"
        elif path_analysis["severity"] > 0.2:
            issues["severity_level"] = "medium"
        else:
            issues["severity_level"] = "low"
        
        return issues
    
    def _generate_recovery_paths(self, cluster: ClusterContainer,
                               original_path: List[str],
                               failed_nodes: List[str]) -> List[List[str]]:
        """Генерация путей восстановления для обхода failed nodes"""
        if not failed_nodes:
            return []
        
        alternative_paths = []
        start_node = original_path[0]
        target_node = original_path[-1]
        
        # Пытаемся найти пути, которые обходят failed nodes
        for failed_node in failed_nodes:
            if failed_node not in original_path:
                continue
            
            # Находим индекс failed node в пути
            try:
                failed_index = original_path.index(failed_node)
            except ValueError:
                continue
            
            # Разбиваем путь на части
            before_failed = original_path[:failed_index]
            after_failed = original_path[failed_index+1:] if failed_index+1 < len(original_path) else []
            
            if not after_failed:
                # Failed node в конце - ищем альтернативный конечный узел
                continue
            
            # Ищем обходные пути для сегмента с failed node
            bypass_paths = self._find_bypass_paths(
                cluster, 
                before_failed[-1] if before_failed else start_node,
                after_failed[0],
                failed_node,
                max_depth=3
            )
            
            for bypass in bypass_paths:
                # Собираем полный путь
                full_path = before_failed + bypass + after_failed
                if full_path not in alternative_paths:
                    alternative_paths.append(full_path)
        
        return alternative_paths
    
    def _find_bypass_paths(self, cluster: ClusterContainer,
                          start: str, target: str,
                          avoid_node: str, max_depth: int) -> List[List[str]]:
        """Поиск путей в обход определенного узла"""
        bypass_paths = []
        
        # Простой BFS с исключением узла
        queue = deque([([start], 0)])
        
        while queue:
            path, depth = queue.popleft()
            current = path[-1]
            
            if current == target:
                bypass_paths.append(path[1:])  # Исключаем start из результата
                continue
            
            if depth >= max_depth:
                continue
            
            # Находим соседей, исключая avoid_node
            for edge_name in cluster.intra_cluster_edges:
                if cluster.has_item(edge_name):
                    edge = cluster.get(edge_name)
                    if isinstance(edge, EdgeEntity) and edge.source_node == current:
                        next_node = edge.target_node
                        if next_node != avoid_node and next_node not in path:
                            new_path = path + [next_node]
                            queue.append((new_path, depth + 1))
        
        return bypass_paths
    
    def _generate_optimization_paths(self, cluster: ClusterContainer,
                                   original_path: List[str],
                                   bottlenecks: List[Dict[str, Any]]) -> List[List[str]]:
        """Генерация оптимизированных путей для обхода bottlenecks"""
        if not bottlenecks:
            return []
        
        alternative_paths = []
        start_node = original_path[0]
        target_node = original_path[-1]
        
        # Для каждого bottleneck пытаемся найти альтернативный путь
        for bottleneck in bottlenecks:
            bottleneck_node = bottleneck["node"]
            bottleneck_index = bottleneck["position"]
            
            # Ищем альтернативные сегменты
            alternative_segments = self._find_alternative_segments(
                cluster,
                original_path[max(0, bottleneck_index-1)],  # Узел перед bottleneck
                original_path[min(len(original_path)-1, bottleneck_index+1)],  # Узел после bottleneck
                bottleneck_node,
                segment_length=2
            )
            
            for segment in alternative_segments:
                # Собираем полный путь
                if bottleneck_index > 0 and bottleneck_index < len(original_path)-1:
                    new_path = (
                        original_path[:bottleneck_index] + 
                        segment + 
                        original_path[bottleneck_index+1:]
                    )
                    if new_path not in alternative_paths:
                        alternative_paths.append(new_path)
        
        return alternative_paths
    
    def _find_alternative_segments(self, cluster: ClusterContainer,
                                 before_node: str, after_node: str,
                                 avoid_node: str, segment_length: int) -> List[List[str]]:
        """Поиск альтернативных сегментов пути"""
        segments = []
        
        # Ищем пути от before_node к after_node, избегая avoid_node
        paths = self._find_all_paths_bfs(cluster, before_node, after_node, segment_length)
        
        for path in paths:
            if avoid_node not in path:
                segments.append(path[1:-1])  # Исключаем начальный и конечный узлы
        
        return segments
    
    def _generate_load_balanced_paths(self, cluster: ClusterContainer,
                                    original_path: List[str],
                                    load_info: Dict[str, float]) -> List[List[str]]:
        """Генерация путей с балансировкой нагрузки"""
        alternative_paths = []
        start_node = original_path[0]
        target_node = original_path[-1]
        
        # Находим наиболее загруженные узлы в пути
        loaded_nodes = []
        for i, node_name in enumerate(original_path[1:-1]):  # Пропускаем первый и последний
            if node_name in load_info and load_info[node_name] > 0.6:
                loaded_nodes.append({
                    "node": node_name,
                    "index": i+1,
                    "load": load_info[node_name]
                })
        
        if not loaded_nodes:
            return []
        
        # Сортируем по нагрузке
        loaded_nodes.sort(key=lambda x: x["load"], reverse=True)
        
        # Для каждого сильно загруженного узла ищем альтернативы
        for loaded_node in loaded_nodes[:2]:  # Обрабатываем только 2 самых загруженных
            alt_paths = self._find_load_balanced_alternatives(
                cluster, original_path, loaded_node, load_info
            )
            alternative_paths.extend(alt_paths)
        
        return alternative_paths
    
    def _find_load_balanced_alternatives(self, cluster: ClusterContainer,
                                       original_path: List[str],
                                       loaded_node: Dict[str, Any],
                                       load_info: Dict[str, float]) -> List[List[str]]:
        """Поиск альтернативных путей с балансировкой нагрузки"""
        alternatives = []
        start = original_path[0]
        target = original_path[-1]
        avoid_node = loaded_node["node"]
        
        # Ищем пути, которые избегают загруженного узла
        alternative_paths = self._find_bypass_paths(cluster, start, target, avoid_node, max_depth=len(original_path)+2)
        
        # Оцениваем нагрузку альтернативных путей
        for path in alternative_paths:
            # Вычисляем среднюю нагрузку пути
            path_loads = []
            for node in path:
                if node in load_info:
                    path_loads.append(load_info[node])
            
            avg_load = np.mean(path_loads) if path_loads else 0.0
            
            # Если альтернативный путь имеет меньшую нагрузку, добавляем его
            if avg_load < loaded_node["load"] * 0.8:  # На 20% меньше нагрузки
                alternatives.append(path)
        
        return alternatives
    
    def _evaluate_alternative_paths(self, cluster: ClusterContainer,
                                  alternative_paths: List[List[str]],
                                  original_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Оценка альтернативных путей"""
        evaluated = []
        
        for path in alternative_paths:
            if len(path) < 2:
                continue
            
            # Анализируем путь
            path_analysis = self._analyze_current_path(cluster, path, [], {})
            
            # Сравниваем с оригинальным путем
            improvement = self._calculate_improvement(original_analysis, path_analysis)
            
            # Оцениваем стоимость переключения
            switching_cost = self._calculate_switching_cost(original_analysis["path_nodes"], path)
            
            evaluated.append({
                "path": path,
                "analysis": path_analysis,
                "improvement_score": improvement,
                "switching_cost": switching_cost,
                "net_benefit": improvement - switching_cost
            })
        
        # Сортируем по net benefit
        evaluated.sort(key=lambda x: x["net_benefit"], reverse=True)
        
        return evaluated
    
    def _calculate_improvement(self, original: Dict[str, Any], 
                             alternative: Dict[str, Any]) -> float:
        """Расчет улучшения альтернативного пути"""
        improvement = 0.0
        
        # Улучшение за счет отсутствия failed nodes
        improvement += len(original["failed_nodes_in_path"]) * 0.3
        
        # Улучшение за счет снижения нагрузки
        orig_bottlenecks = len(original["bottlenecks"])
        alt_bottlenecks = len(alternative["bottlenecks"])
        improvement += (orig_bottlenecks - alt_bottlenecks) * 0.2
        
        # Улучшение за счет здоровья пути
        health_improvement = alternative["overall_health"] - original["overall_health"]
        improvement += max(0, health_improvement) * 0.4
        
        # Штраф за увеличение длины пути
        length_penalty = max(0, alternative["path_length"] - original["path_length"]) * 0.1
        improvement -= length_penalty
        
        return max(0, improvement)
    
    def _calculate_switching_cost(self, original_path: List[str],
                                new_path: List[str]) -> float:
        """Расчет стоимости переключения на новый путь"""
        # Базовая стоимость переключения
        base_cost = 0.1
        
        # Дополнительная стоимость за различие в путях
        common_nodes = len(set(original_path) & set(new_path))
        total_nodes = len(set(original_path) | set(new_path))
        
        similarity = common_nodes / max(1, total_nodes)
        dissimilarity_cost = (1.0 - similarity) * 0.2
        
        return base_cost + dissimilarity_cost
    
    def _apply_rerouting(self, cluster: ClusterContainer,
                        original_path: List[str],
                        best_alternative: Optional[Dict[str, Any]],
                        issues_detected: Dict[str, List]) -> Dict[str, Any]:
        """Применение перемаршрутизации"""
        if not best_alternative or best_alternative["net_benefit"] < 0.1:
            return {
                "applied": False,
                "reason": "No beneficial alternative found",
                "original_path_used": True
            }
        
        new_path = best_alternative["path"]
        
        # Регистрируем решение о перемаршрутизации
        rerouting_decision = {
            "timestamp": datetime.now(),
            "cluster": cluster.name,
            "from_path": original_path,
            "to_path": new_path,
            "improvement_expected": best_alternative["improvement_score"],
            "net_benefit": best_alternative["net_benefit"],
            "issues_resolved": {
                "failures": len(issues_detected["failures"]),
                "bottlenecks": len(issues_detected["bottlenecks"]),
                "severity": issues_detected["severity_level"]
            }
        }
        
        # В реальной реализации здесь было бы обновление состояния маршрутизации
        # и уведомление заинтересованных компонентов
        
        return {
            "applied": True,
            "new_path": new_path,
            "improvement_expected": best_alternative["improvement_score"],
            "decision_recorded": True,
            "rerouting_decision": rerouting_decision
        }
    
    # Вспомогательные методы для _routing_meta_learning
    def _analyze_routing_patterns(self, routing_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Анализ паттернов маршрутизации из истории"""
        if not routing_history:
            return {"patterns_found": 0, "common_paths": [], "failure_patterns": []}
        
        patterns = {
            "total_routes": len(routing_history),
            "common_paths": [],
            "failure_patterns": [],
            "performance_trends": {},
            "node_popularity": defaultdict(int)
        }
        
        # Анализируем популярные пути
        path_counts = defaultdict(int)
        for record in routing_history:
            if "path_length" in record:
                # Группируем по длине пути
                length = record["path_length"]
                path_counts[length] += 1
            
            # Анализируем популярность узлов
            if "optimal_path" in record and isinstance(record["optimal_path"], list):
                for node in record["optimal_path"]:
                    patterns["node_popularity"][node] += 1
        
        # Находим наиболее частые длины путей
        if path_counts:
            common_lengths = sorted(path_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            patterns["common_paths"] = [
                {"length": length, "count": count, "frequency": count/len(routing_history)}
                for length, count in common_lengths
            ]
        
        # Анализируем паттерны сбоев из _failure_patterns
        if hasattr(self, '_failure_patterns') and self._failure_patterns:
            failure_nodes = sorted(
                self._failure_patterns.items(),
                key=lambda x: x[1]["count"],
                reverse=True
            )[:5]
            
            patterns["failure_patterns"] = [
                {
                    "node": node,
                    "failure_count": data["count"],
                    "last_failure": data["last_failure"],
                    "frequency": data["count"] / len(routing_history)
                }
                for node, data in failure_nodes
            ]
        
        # Анализируем тренды производительности
        if len(routing_history) > 10:
            recent = routing_history[-10:]
            older = routing_history[:10]
            
            recent_confidence = np.mean([r.get("confidence", 0) for r in recent])
            older_confidence = np.mean([r.get("confidence", 0) for r in older])
            
            patterns["performance_trends"] = {
                "confidence_trend": recent_confidence - older_confidence,
                "recent_avg_confidence": recent_confidence,
                "older_avg_confidence": older_confidence
            }
        
        patterns["patterns_found"] = len(patterns["common_paths"]) + len(patterns["failure_patterns"])
        
        return patterns
    
    def _extract_routing_heuristics(self, pattern_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Извлечение heuristics из паттернов маршрутизации"""
        heuristics = []
        
        # Heuristic 1: Избегание частых сбоев
        for failure_pattern in pattern_analysis.get("failure_patterns", []):
            if failure_pattern["frequency"] > 0.1:  # Частые сбои
                heuristics.append({
                    "type": "avoidance",
                    "target": failure_pattern["node"],
                    "strength": min(1.0, failure_pattern["frequency"] * 2),
                    "reason": f"Frequent failures ({failure_pattern['failure_count']} times)"
                })
        
        # Heuristic 2: Предпочтение популярных длин путей
        for common_path in pattern_analysis.get("common_paths", []):
            if common_path["frequency"] > 0.2:  # Частая длина
                heuristics.append({
                    "type": "preference",
                    "preference": "path_length",
                    "value": common_path["length"],
                    "strength": common_path["frequency"],
                    "reason": f"Common path length ({common_path['count']} occurrences)"
                })
        
        # Heuristic 3: Предпочтение популярных узлов
        node_popularity = pattern_analysis.get("node_popularity", {})
        if node_popularity:
            total_routes = pattern_analysis.get("total_routes", 1)
            popular_nodes = sorted(
                node_popularity.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
            
            for node, count in popular_nodes:
                frequency = count / total_routes
                if frequency > 0.3:
                    heuristics.append({
                        "type": "preference",
                        "preference": "node",
                        "value": node,
                        "strength": frequency,
                        "reason": f"Popular node ({count} appearances)"
                    })
        
        # Heuristic 4: На основе трендов производительности
        trends = pattern_analysis.get("performance_trends", {})
        if trends.get("confidence_trend", 0) < -0.1:
            heuristics.append({
                "type": "adjustment",
                "adjustment": "increase_search_iterations",
                "value": 20,
                "strength": abs(trends["confidence_trend"]),
                "reason": "Declining confidence trend"
            })
        
        return heuristics
    
    def _test_generalization(self, cluster: ClusterContainer,
                           heuristics: List[Dict[str, Any]],
                           test_cases: Dict[str, Any]) -> Dict[str, Any]:
        """Тестирование обобщения heuristics на новые запросы"""
        test_results = {
            "total_tests": 0,
            "passed_tests": 0,
            "improvement_shown": 0,
            "average_improvement": 0.0
        }
        
        if not test_cases or "cases" not in test_cases:
            return test_results
        
        improvements = []
        
        for test_case in test_cases["cases"][:5]:  # Тестируем только 5 случаев
            test_results["total_tests"] += 1
            
            # Применяем heuristics к тестовому случаю
            enhanced_result = self._apply_heuristics_to_case(cluster, test_case, heuristics)
            
            # Сравниваем с базовым результатом
            if "baseline" in test_case and "enhanced" in enhanced_result:
                baseline = test_case["baseline"].get("score", 0)
                enhanced = enhanced_result["enhanced"].get("score", 0)
                
                if enhanced > baseline:
                    test_results["passed_tests"] += 1
                    test_results["improvement_shown"] += 1
                    improvement = (enhanced - baseline) / max(0.001, baseline)
                    improvements.append(improvement)
        
        if improvements:
            test_results["average_improvement"] = np.mean(improvements)
        
        return test_results
    
    def _apply_heuristics_to_case(self, cluster: ClusterContainer,
                                test_case: Dict[str, Any],
                                heuristics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Применение heuristics к тестовому случаю"""
        # Упрощенная реализация: модифицируем параметры поиска на основе heuristics
        modified_attributes = test_case.get("attributes", {}).copy()
        
        for heuristic in heuristics:
            if heuristic["type"] == "avoidance":
                # Добавляем узлы для избегания
                if "avoid_nodes" not in modified_attributes:
                    modified_attributes["avoid_nodes"] = []
                modified_attributes["avoid_nodes"].append(heuristic["target"])
            
            elif heuristic["type"] == "adjustment" and heuristic["adjustment"] == "increase_search_iterations":
                # Увеличиваем итерации поиска
                modified_attributes["search_iterations"] = (
                    modified_attributes.get("search_iterations", 100) + heuristic["value"]
                )
        
        # Выполняем предсказание с модифицированными атрибутами
        # В реальной реализации здесь был бы вызов _routing_predict_optimal_path
        enhanced_result = {
            "enhanced": {
                "score": 0.7,  # Заглушка
                "attributes_modified": len(heuristics)
            }
        }
        
        return enhanced_result
    
    def _evolve_routing_heuristics(self, learned_heuristics: List[Dict[str, Any]],
                                 pattern_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Эволюционная оптимизация heuristics"""
        if not learned_heuristics:
            return []
        
        optimized_heuristics = []
        
        for heuristic in learned_heuristics:
            # Создаем модифицированную версию heuristic
            optimized = heuristic.copy()
            
            # Эволюционные операторы
            evolution_operator = random.choice(["mutate_strength", "combine", "specialize"])
            
            if evolution_operator == "mutate_strength":
                # Мутация силы heuristic
                mutation = random.uniform(-0.2, 0.2)
                optimized["strength"] = max(0.1, min(1.0, heuristic["strength"] + mutation))
                optimized["evolution_note"] = f"Strength mutated by {mutation:.3f}"
            
            elif evolution_operator == "combine" and len(learned_heuristics) > 1:
                # Комбинация с другой heuristic
                other = random.choice([h for h in learned_heuristics if h != heuristic])
                if heuristic["type"] == other["type"]:
                    optimized["strength"] = (heuristic["strength"] + other["strength"]) / 2
                    optimized["evolution_note"] = f"Combined with similar heuristic"
            
            elif evolution_operator == "specialize":
                # Специализация heuristic
                if "strength" in optimized:
                    optimized["strength"] = min(1.0, heuristic["strength"] * 1.3)
                    optimized["evolution_note"] = "Specialized (increased strength)"
            
            # Добавляем информацию о эволюции
            if "evolution_note" not in optimized:
                optimized["evolution_note"] = "No evolution applied"
            
            optimized_heuristics.append(optimized)
        
        # Отбираем лучшие heuristics
        if len(optimized_heuristics) > 10:
            # Сортируем по силе и отбираем топ-10
            optimized_heuristics.sort(key=lambda h: h.get("strength", 0), reverse=True)
            optimized_heuristics = optimized_heuristics[:10]
        
        return optimized_heuristics
    
    def _update_routing_models(self, cluster: ClusterContainer,
                             optimized_heuristics: List[Dict[str, Any]],
                             pattern_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Обновление моделей маршрутизации на основе оптимизированных heuristics"""
        # В реальной реализации здесь было бы обновление весов моделей, параметров и т.д.
        # В данной реализации просто регистрируем обновление
        
        update_summary = {
            "models_updated": 1,  # Основная модель маршрутизации
            "heuristics_integrated": len(optimized_heuristics),
            "cache_cleared": False,
            "expected_improvement": 0.0
        }
        
        # Очищаем кэш, если есть значительные изменения
        if optimized_heuristics and pattern_analysis.get("patterns_found", 0) > 5:
            self._path_cache.clear()
            update_summary["cache_cleared"] = True
        
        # Оцениваем ожидаемое улучшение
        if optimized_heuristics:
            avg_strength = np.mean([h.get("strength", 0) for h in optimized_heuristics])
            update_summary["expected_improvement"] = min(0.3, avg_strength * 0.2)
        
        return update_summary
