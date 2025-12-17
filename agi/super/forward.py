"""
AGI Super Classes for MSB Architecture - Реализация обработчиков для AGI системы
"""

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


class ForwardPropagationSuper(Super):
    """
    Распространение активации в графовой архитектуре AGI.
    
    Реализует различные стратегии распространения активации:
    - Параллельная breadth-first активация
    - Attention-based гейтирование
    - Временная интеграция
    """
    
    OPERATION = "forward"
    
    def __init__(self, manipulator=None, methods=None, cache_size=2048):
        super().__init__(manipulator, methods, cache_size)
        self._attention_weights = {}
        self._phase_synchronization = {}
        self._executor = ThreadPoolExecutor(max_workers=4)
    
    def _forward_node(self, node: NodeEntity, attributes: Dict[str, Any]) -> Dict[str, Any]:
        """Прямое распространение через отдельный узел"""
        input_vector = attributes.get("input_vector")
        use_memory = attributes.get("use_memory", True)
        
        if input_vector is None:
            return self._build_response(node, False, None, None, "No input_vector provided")
        
        result = node.forward(input_vector=input_vector, use_memory=use_memory)
        return self._build_response(node, True, "forward", result)
    
    def _forward_cluster(self, cluster: ClusterContainer, attributes: Dict[str, Any]) -> Dict[str, Any]:
        """Распространение через кластер с выбором стратегии"""
        strategy = attributes.get("strategy", "parallel_breadth_first")
        
        if strategy == "parallel_breadth_first":
            return self._forward_parallel_breadth_first(cluster, attributes)
        elif strategy == "attention_gating":
            return self._forward_attention_gating(cluster, attributes)
        elif strategy == "temporal_integration":
            return self._forward_temporal_integration(cluster, attributes)
        else:
            # Используем стандартное распространение кластера
            return self._forward_default(cluster, attributes)
    
    def _forward_parallel_breadth_first(self, cluster: ClusterContainer, 
                                      attributes: Dict[str, Any]) -> Dict[str, Any]:
        """
        Параллельная breadth-first активация с message passing.
        
        Активирует узлы на каждом уровне графа параллельно,
        передает сообщения между уровнями с attention механизмом.
        """
        if not cluster.isactive:
            return self._build_response(cluster, False, None, None, "Cluster is inactive")
        
        input_signals = attributes.get("input_signals", {})
        max_depth = attributes.get("max_depth", 3)
        energy_budget = attributes.get("energy_budget", 1.0)
        
        start_time = datetime.now()
        activated_nodes = {}
        level_results = {}
        
        # Шаг 1: Активация входных узлов (уровень 0)
        level_0_nodes = self._get_input_nodes(cluster, input_signals)
        level_0_results = self._activate_nodes_parallel(cluster, level_0_nodes, input_signals)
        
        activated_nodes[0] = level_0_results["activated"]
        level_results[0] = level_0_results
        
        # Шаг 2: Breadth-first распространение
        for level in range(1, max_depth + 1):
            if not activated_nodes[level - 1]:
                break
            
            # Собираем узлы для активации на текущем уровне
            current_level_nodes = self._get_next_level_nodes(
                cluster, activated_nodes[level - 1]
            )
            
            if not current_level_nodes:
                break
            
            # Вычисляем входные сигналы для текущего уровня
            level_inputs = self._aggregate_messages(
                cluster, activated_nodes[level - 1], current_level_nodes
            )
            
            # Параллельная активация
            level_result = self._activate_nodes_parallel(
                cluster, current_level_nodes, level_inputs
            )
            
            activated_nodes[level] = level_result["activated"]
            level_results[level] = level_result
        
        # Шаг 3: Интеграция результатов
        final_output = self._integrate_level_results(level_results)
        total_activated = sum(len(nodes) for nodes in activated_nodes.values())
        
        # Эмерджентная синхронизация фаз
        synchronization_score = self._calculate_phase_synchronization(
            cluster, activated_nodes
        )
        
        latency = (datetime.now() - start_time).total_seconds()
        
        result = {
            "status": "success",
            "cluster": cluster.name,
            "total_activated_nodes": total_activated,
            "levels_activated": len(activated_nodes),
            "final_activation": final_output,
            "synchronization_score": synchronization_score,
            "latency_seconds": latency,
            "level_details": level_results
        }
        
        logger.info(f"Parallel BFS completed for cluster '{cluster.name}': "
                   f"{total_activated} nodes across {len(activated_nodes)} levels")
        
        return self._build_response(cluster, True, "parallel_breadth_first", result)
    
    def _forward_attention_gating(self, cluster: ClusterContainer, 
                                attributes: Dict[str, Any]) -> Dict[str, Any]:
        """
        Attention-based гейтирование потока информации.
        
        Вычисляет relevance score для каждого узла,
        динамически гейтирует поток информации,
        выполняет adaptive pruning малозначимых путей.
        """
        if not cluster.isactive:
            return self._build_response(cluster, False, None, None, "Cluster is inactive")
        
        input_signals = attributes.get("input_signals", {})
        attention_threshold = attributes.get("attention_threshold", 0.3)
        pruning_ratio = attributes.get("pruning_ratio", 0.5)
        
        start_time = datetime.now()
        
        # Шаг 1: Вычисление relevance scores
        relevance_scores = self._calculate_relevance_scores(cluster, input_signals)
        
        # Шаг 2: Attention-based гейтирование
        gated_nodes = self._apply_attention_gating(
            cluster, relevance_scores, attention_threshold
        )
        
        # Шаг 3: Adaptive pruning
        pruned_nodes = self._adaptive_pruning(
            cluster, gated_nodes, pruning_ratio
        )
        
        # Шаг 4: Активация через отобранные узлы
        activation_result = cluster.propagate_activation(
            input_signals={node: 1.0 for node in pruned_nodes},
            max_depth=2,
            energy_budget=0.8
        )
        
        # Шаг 5: Анализ эффективности
        efficiency_metrics = self._calculate_attention_efficiency(
            cluster, gated_nodes, pruned_nodes, activation_result
        )
        
        latency = (datetime.now() - start_time).total_seconds()
        
        result = {
            "status": "success",
            "cluster": cluster.name,
            "total_nodes": len(relevance_scores),
            "gated_nodes": len(gated_nodes),
            "pruned_nodes": len(pruned_nodes),
            "attention_efficiency": efficiency_metrics,
            "activation_result": activation_result,
            "latency_seconds": latency
        }
        
        logger.info(f"Attention gating completed for cluster '{cluster.name}': "
                   f"{len(pruned_nodes)}/{len(relevance_scores)} nodes activated")
        
        return self._build_response(cluster, True, "attention_gating", result)
    
    def _forward_temporal_integration(self, cluster: ClusterContainer, 
                                    attributes: Dict[str, Any]) -> Dict[str, Any]:
        """
        Обработка временных последовательностей с интеграцией окон.
        
        Интегрирует информацию из окон различной длительности,
        генерирует prediction-error сигналы для обучения.
        """
        if not cluster.isactive:
            return self._build_response(cluster, False, None, None, "Cluster is inactive")
        
        temporal_sequence = attributes.get("temporal_sequence", [])
        window_sizes = attributes.get("window_sizes", [1, 3, 5])
        
        start_time = datetime.now()
        
        # Шаг 1: Создание временных окон
        windows = self._create_temporal_windows(temporal_sequence, window_sizes)
        
        # Шаг 2: Параллельная обработка окон
        window_results = []
        for window in windows:
            window_result = cluster.propagate_activation(
                input_signals=window,
                max_depth=2,
                energy_budget=0.6
            )
            window_results.append(window_result)
        
        # Шаг 3: Временная интеграция
        integrated_result = self._temporal_integration(window_results)
        
        # Шаг 4: Генерация prediction-error
        prediction_errors = self._calculate_prediction_errors(
            window_results, integrated_result
        )
        
        # Шаг 5: Обновление внутренних состояний
        self._update_temporal_states(cluster, window_results, prediction_errors)
        
        latency = (datetime.now() - start_time).total_seconds()
        
        result = {
            "status": "success",
            "cluster": cluster.name,
            "sequence_length": len(temporal_sequence),
            "windows_processed": len(window_results),
            "integrated_activation": integrated_result,
            "prediction_errors": prediction_errors,
            "temporal_coherence": self._calculate_temporal_coherence(window_results),
            "latency_seconds": latency
        }
        
        logger.info(f"Temporal integration completed for cluster '{cluster.name}': "
                   f"{len(window_results)} windows processed")
        
        return self._build_response(cluster, True, "temporal_integration", result)
    
    def _forward_default(self, cluster: ClusterContainer, 
                        attributes: Dict[str, Any]) -> Dict[str, Any]:
        """Стандартное распространение через кластер"""
        input_signals = attributes.get("input_signals", {})
        max_depth = attributes.get("max_depth", 3)
        energy_budget = attributes.get("energy_budget", 1.0)
        
        result = cluster.propagate_activation(
            input_signals=input_signals,
            max_depth=max_depth,
            energy_budget=energy_budget
        )
        
        return self._build_response(cluster, True, "propagate_activation", result)
    
    # Вспомогательные методы
    def _get_input_nodes(self, cluster: ClusterContainer, 
                        input_signals: Dict[str, float]) -> List[str]:
        """Получить узлы, соответствующие входным сигналам"""
        input_nodes = []
        for node_name in input_signals.keys():
            if cluster.has_item(node_name):
                node = cluster.get(node_name)
                if isinstance(node, NodeEntity) and node.isactive:
                    input_nodes.append(node_name)
        return input_nodes
    
    def _activate_nodes_parallel(self, cluster: ClusterContainer, 
                               node_names: List[str], 
                               input_signals: Dict[str, float]) -> Dict[str, Any]:
        """Параллельная активация узлов"""
        activated = []
        activations = {}
        total_energy = 0.0
        
        def activate_single_node(node_name):
            if node_name in input_signals:
                node = cluster.get(node_name)
                if isinstance(node, NodeEntity):
                    result = node.forward(
                        input_vector=np.array([input_signals[node_name]], dtype=np.float32),
                        use_memory=True
                    )
                    if result.get("status") != "inactive" and result.get("confidence", 0) > 0.3:
                        return node_name, result.get("confidence", 0), 0.01
            return None, 0.0, 0.0
        
        # Простая реализация параллелизма (можно заменить на реальный ThreadPool)
        for node_name in node_names:
            node_name, confidence, energy = activate_single_node(node_name)
            if node_name:
                activated.append(node_name)
                activations[node_name] = confidence
                total_energy += energy
        
        return {
            "activated": activated,
            "activations": activations,
            "total_energy": total_energy,
            "average_confidence": np.mean(list(activations.values())) if activations else 0.0
        }
    
    def _get_next_level_nodes(self, cluster: ClusterContainer, 
                            previous_nodes: List[str]) -> List[str]:
        """Получить узлы следующего уровня на основе связей"""
        next_level = set()
        
        for node_name in previous_nodes:
            # Найти все исходящие связи
            for edge_name in cluster.intra_cluster_edges:
                if cluster.has_item(edge_name):
                    edge = cluster.get(edge_name)
                    if isinstance(edge, EdgeEntity) and edge.source_node == node_name:
                        next_level.add(edge.target_node)
        
        return list(next_level)
    
    def _aggregate_messages(self, cluster: ClusterContainer,
                          source_nodes: List[str],
                          target_nodes: List[str]) -> Dict[str, float]:
        """Агрегация сообщений между узлами"""
        messages = {node: 0.0 for node in target_nodes}
        
        for source in source_nodes:
            for edge_name in cluster.intra_cluster_edges:
                if cluster.has_item(edge_name):
                    edge = cluster.get(edge_name)
                    if isinstance(edge, EdgeEntity) and edge.source_node == source:
                        if edge.target_node in messages:
                            # Вес сообщения зависит от веса связи и активности источника
                            messages[edge.target_node] += edge.weight * 0.5
        
        return messages
    
    def _integrate_level_results(self, level_results: Dict[int, Dict[str, Any]]) -> float:
        """Интеграция результатов с разных уровней"""
        if not level_results:
            return 0.0
        
        total_activation = 0.0
        total_weight = 0.0
        
        for level, result in level_results.items():
            weight = 1.0 / (level + 1)  # Более высокие уровни имеют меньший вес
            total_activation += result.get("average_confidence", 0.0) * weight
            total_weight += weight
        
        return total_activation / total_weight if total_weight > 0 else 0.0
    
    def _calculate_phase_synchronization(self, cluster: ClusterContainer,
                                       activated_nodes: Dict[int, List[str]]) -> float:
        """Расчет синхронизации фаз активации"""
        if not activated_nodes:
            return 0.0
        
        # Простая метрика синхронизации
        activation_counts = [len(nodes) for nodes in activated_nodes.values()]
        if len(activation_counts) < 2:
            return 1.0
        
        # Коэффициент вариации (ниже = лучше синхронизация)
        mean_count = np.mean(activation_counts)
        if mean_count == 0:
            return 0.0
        
        std_count = np.std(activation_counts)
        cv = std_count / mean_count
        
        # Преобразуем в оценку синхронизации (1.0 = идеальная синхронизация)
        return max(0.0, 1.0 - cv)
    
    def _calculate_relevance_scores(self, cluster: ClusterContainer,
                                  input_signals: Dict[str, float]) -> Dict[str, float]:
        """Вычисление relevance scores для узлов"""
        relevance_scores = {}
        
        for item in cluster.get_active_items():
            if isinstance(item, NodeEntity):
                # Базовая релевантность на основе специализации и входных сигналов
                score = 0.0
                
                # Проверка соответствия специализации
                if hasattr(item, 'specialization'):
                    # Упрощенная проверка (в реальности - векторное сравнение)
                    score += 0.3
                
                # Проверка наличия входных сигналов
                if item.name in input_signals:
                    score += 0.4
                
                # Учет предыдущей активности
                if item.activation_count > 0:
                    score += min(0.3, item.activation_count / 100)
                
                relevance_scores[item.name] = score
        
        return relevance_scores
    
    def _apply_attention_gating(self, cluster: ClusterContainer,
                              relevance_scores: Dict[str, float],
                              threshold: float) -> List[str]:
        """Применение attention гейтирования"""
        gated_nodes = []
        
        for node_name, score in relevance_scores.items():
            if score >= threshold:
                gated_nodes.append(node_name)
        
        return gated_nodes
    
    def _adaptive_pruning(self, cluster: ClusterContainer,
                        gated_nodes: List[str],
                        pruning_ratio: float) -> List[str]:
        """Adaptive pruning малозначимых путей"""
        if not gated_nodes:
            return []
        
        # Сортируем узлы по relevance
        node_scores = []
        for node_name in gated_nodes:
            node = cluster.get(node_name)
            if isinstance(node, NodeEntity):
                # Комбинированная оценка: relevance + utility
                utility = node.activation_count / max(1, node.activation_count + 10)
                score = utility * 0.7 + 0.3  # Базовая оценка
                node_scores.append((score, node_name))
        
        # Сортируем по убыванию оценки
        node_scores.sort(reverse=True)
        
        # Выбираем топ-N узлов
        keep_count = max(1, int(len(gated_nodes) * (1 - pruning_ratio)))
        pruned_nodes = [node_name for _, node_name in node_scores[:keep_count]]
        
        return pruned_nodes
    
    def _calculate_attention_efficiency(self, cluster: ClusterContainer,
                                      gated_nodes: List[str],
                                      pruned_nodes: List[str],
                                      activation_result: Dict[str, Any]) -> Dict[str, float]:
        """Расчет эффективности attention механизма"""
        total_nodes = len([item for item in cluster.get_items() if isinstance(item, NodeEntity)])
        
        efficiency_metrics = {
            "gating_efficiency": len(pruned_nodes) / max(1, len(gated_nodes)),
            "resource_saving": 1.0 - (len(pruned_nodes) / max(1, total_nodes)),
            "activation_success_rate": activation_result.get("status") == "success",
            "energy_per_node": activation_result.get("energy_used", 0.0) / max(1, len(pruned_nodes))
        }
        
        return efficiency_metrics
    
    def _create_temporal_windows(self, temporal_sequence: List[Dict[str, float]],
                               window_sizes: List[int]) -> List[Dict[str, float]]:
        """Создание временных окон из последовательности"""
        windows = []
        sequence_length = len(temporal_sequence)
        
        for window_size in window_sizes:
            for i in range(sequence_length - window_size + 1):
                window = {}
                for j in range(window_size):
                    time_step = i + j
                    if time_step < sequence_length:
                        # Агрегируем сигналы в окне
                        for key, value in temporal_sequence[time_step].items():
                            if key not in window:
                                window[key] = 0.0
                            window[key] += value / window_size
                if window:
                    windows.append(window)
        
        return windows
    
    def _temporal_integration(self, window_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Интеграция результатов временных окон"""
        if not window_results:
            return {"integrated_activation": 0.0, "confidence": 0.0}
        
        total_activation = 0.0
        total_confidence = 0.0
        count = 0
        
        for result in window_results:
            if result.get("status") == "success":
                total_activation += result.get("output_activation", 0.0)
                # Используем coherence как меру confidence
                total_confidence += result.get("coherence", 0.0)
                count += 1
        
        if count == 0:
            return {"integrated_activation": 0.0, "confidence": 0.0}
        
        return {
            "integrated_activation": total_activation / count,
            "confidence": total_confidence / count,
            "windows_integrated": count
        }
    
    def _calculate_prediction_errors(self, window_results: List[Dict[str, Any]],
                                   integrated_result: Dict[str, Any]) -> List[float]:
        """Расчет prediction errors"""
        errors = []
        target_activation = integrated_result.get("integrated_activation", 0.0)
        
        for result in window_results:
            if result.get("status") == "success":
                activation = result.get("output_activation", 0.0)
                error = abs(activation - target_activation)
                errors.append(error)
        
        return errors
    
    def _update_temporal_states(self, cluster: ClusterContainer,
                              window_results: List[Dict[str, Any]],
                              prediction_errors: List[float]):
        """Обновление временных состояний на основе prediction errors"""
        # Простая реализация: обновляем energy reserve на основе ошибок
        avg_error = np.mean(prediction_errors) if prediction_errors else 0.0
        
        # Чем выше ошибка, тем больше энергии потребуется для коррекции
        energy_adjustment = min(0.1, avg_error * 0.05)
        cluster.energy_reserve = max(0.0, cluster.energy_reserve - energy_adjustment)
    
    def _calculate_temporal_coherence(self, window_results: List[Dict[str, Any]]) -> float:
        """Расчет временной когерентности"""
        if len(window_results) < 2:
            return 1.0
        
        activations = []
        for result in window_results:
            if result.get("status") == "success":
                activations.append(result.get("output_activation", 0.0))
        
        if len(activations) < 2:
            return 0.0
        
        # Рассчитываем автокорреляцию с лагом 1
        correlation = np.corrcoef(activations[:-1], activations[1:])[0, 1]
        
        # Преобразуем корреляцию в когерентность [0, 1]
        return (correlation + 1) / 2 if not np.isnan(correlation) else 0.0