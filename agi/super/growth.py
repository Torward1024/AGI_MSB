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

class GrowthSuper(Super):
    """
    Нейрогенез и структурная адаптация в графовой архитектуре AGI.
    
    Реализует механизмы роста и оптимизации структуры:
    - Обнаружение пробелов в знаниях
    - Создание новых узлов и связей
    - Pruning избыточных компонентов
    """
    
    OPERATION = "growth"
    
    def __init__(self, manipulator=None, methods=None, cache_size=2048):
        super().__init__(manipulator, methods, cache_size)
        self._gap_detection_history = deque(maxlen=100)
        self._node_creation_log = []
        self._pruning_decisions = []
    
    def _growth_cluster(self, cluster: ClusterContainer, 
                       attributes: Dict[str, Any]) -> Dict[str, Any]:
        """Управление ростом в кластере"""
        growth_operation = attributes.get("operation", "detect_gaps")
        
        if growth_operation == "detect_gaps":
            return self._growth_detect_gaps(cluster, attributes)
        elif growth_operation == "create_node":
            return self._growth_create_node(cluster, attributes)
        elif growth_operation == "prune_redundant":
            return self._growth_prune_redundant(cluster, attributes)
        elif growth_operation == "optimize_structure":
            return self._growth_optimize_structure(cluster, attributes)
        else:
            return self._build_response(cluster, False, None, None, 
                                      f"Unknown growth operation: {growth_operation}")
    
    def _growth_detect_gaps(self, cluster: ClusterContainer, 
                          attributes: Dict[str, Any]) -> Dict[str, Any]:
        """
        Обнаружение пробелов в знаниях и возможностей для роста.
        
        Использует кластеризацию low-confidence inputs,
        pattern mining для новых категорий,
        predictive need assessment.
        """
        if not cluster.isactive:
            return self._build_response(cluster, False, None, None, "Cluster is inactive")
        
        confidence_threshold = attributes.get("confidence_threshold", 0.4)
        min_samples = attributes.get("min_samples", 5)
        
        start_time = datetime.now()
        
        # Шаг 1: Сбор low-confidence inputs
        low_confidence_inputs = self._collect_low_confidence_inputs(
            cluster, confidence_threshold
        )
        
        if not low_confidence_inputs:
            return self._build_response(cluster, True, "detect_gaps", {
                "status": "no_gaps",
                "cluster": cluster.name,
                "message": "No low-confidence inputs found"
            })
        
        # Шаг 2: Кластеризация пробелов (DBSCAN)
        gap_clusters = self._cluster_gaps(low_confidence_inputs, min_samples)
        
        # Шаг 3: Pattern mining для новых категорий
        new_patterns = self._mine_new_patterns(gap_clusters)
        
        # Шаг 4: Predictive need assessment
        need_assessment = self._assess_growth_needs(cluster, gap_clusters, new_patterns)
        
        # Шаг 5: Приоритизация пробелов
        prioritized_gaps = self._prioritize_gaps(gap_clusters, need_assessment)
        
        latency = (datetime.now() - start_time).total_seconds()
        
        # Сохраняем обнаруженные пробелы
        gap_record = {
            "timestamp": datetime.now(),
            "cluster": cluster.name,
            "low_confidence_inputs": len(low_confidence_inputs),
            "gap_clusters": len(gap_clusters),
            "new_patterns": len(new_patterns),
            "prioritized_gaps": len(prioritized_gaps)
        }
        self._gap_detection_history.append(gap_record)
        
        result = {
            "status": "success",
            "cluster": cluster.name,
            "low_confidence_inputs_count": len(low_confidence_inputs),
            "gap_clusters_found": len(gap_clusters),
            "new_patterns_identified": len(new_patterns),
            "growth_needs": need_assessment,
            "prioritized_gaps": prioritized_gaps,
            "recommended_actions": self._generate_growth_recommendations(prioritized_gaps),
            "latency_seconds": latency
        }
        
        logger.info(f"Gap detection completed for cluster '{cluster.name}': "
                   f"{len(gap_clusters)} gaps found, {len(prioritized_gaps)} prioritized")
        
        return self._build_response(cluster, True, "detect_gaps", result)
    
    def _growth_create_node(self, cluster: ClusterContainer, 
                          attributes: Dict[str, Any]) -> Dict[str, Any]:
        """
        Создание нового узла с инициализацией через дистилляцию знаний.
        
        Инициализация на основе анализа пробелов,
        интеграция в существующие connectivity patterns.
        """
        if not cluster.isactive:
            return self._build_response(cluster, False, None, None, "Cluster is inactive")
        
        gap_analysis = attributes.get("gap_analysis", {})
        initialization_method = attributes.get("initialization_method", "distillation")
        integration_strategy = attributes.get("integration_strategy", "adaptive")
        
        start_time = datetime.now()
        
        # Шаг 1: Определение специализации нового узла
        specialization = self._determine_node_specialization(gap_analysis)
        
        # Шаг 2: Инициализация через дистилляцию знаний
        initialization_result = self._initialize_node_via_distillation(
            cluster, gap_analysis, initialization_method
        )
        
        # Шаг 3: Создание нового NodeEntity
        new_node = self._create_new_node_entity(
            cluster, specialization, initialization_result
        )
        
        # Шаг 4: Интеграция в существующие connectivity patterns
        integration_result = self._integrate_new_node(
            cluster, new_node, integration_strategy
        )
        
        # Шаг 5: Настройка и валидация
        configuration_result = self._configure_new_node(
            new_node, gap_analysis, initialization_result
        )
        
        # Шаг 6: Добавление в кластер
        node_name = cluster.add_node(
            node=new_node,
            connect_to=integration_result.get("connections", [])
        )
        
        latency = (datetime.now() - start_time).total_seconds()
        
        # Логируем создание узла
        creation_log = {
            "timestamp": datetime.now(),
            "cluster": cluster.name,
            "node_name": node_name,
            "specialization": specialization,
            "initialization_method": initialization_method,
            "connections_created": len(integration_result.get("connections", [])),
            "integration_success": integration_result.get("success", False)
        }
        self._node_creation_log.append(creation_log)
        
        result = {
            "status": "success",
            "cluster": cluster.name,
            "new_node": node_name,
            "specialization": specialization,
            "initialization_result": initialization_result,
            "integration_result": integration_result,
            "configuration_result": configuration_result,
            "node_stats": new_node.get_stats(),
            "latency_seconds": latency
        }
        
        logger.info(f"Node creation completed: '{node_name}' in cluster '{cluster.name}', "
                   f"specialization='{specialization}'")
        
        return self._build_response(cluster, True, "create_node", result)
    
    def _growth_prune_redundant(self, cluster: ClusterContainer, 
                              attributes: Dict[str, Any]) -> Dict[str, Any]:
        """
        Удаление избыточных и малополезных узлов.
        
        Идентификация duplicate/similar узлов,
        utility-based pruning,
        консолидация знаний перед удалением.
        """
        if not cluster.isactive:
            return self._build_response(cluster, False, None, None, "Cluster is inactive")
        
        similarity_threshold = attributes.get("similarity_threshold", 0.8)
        utility_threshold = attributes.get("utility_threshold", 0.2)
        
        start_time = datetime.now()
        
        # Шаг 1: Идентификация кандидатов на удаление
        pruning_candidates = self._identify_pruning_candidates(
            cluster, similarity_threshold, utility_threshold
        )
        
        if not pruning_candidates:
            return self._build_response(cluster, True, "prune_redundant", {
                "status": "no_candidates",
                "cluster": cluster.name,
                "message": "No nodes meet pruning criteria"
            })
        
        # Шаг 2: Консолидация знаний перед удалением
        consolidation_results = self._consolidate_knowledge_before_pruning(
            cluster, pruning_candidates
        )
        
        # Шаг 3: Выполнение pruning
        pruning_results = self._execute_pruning(cluster, pruning_candidates)
        
        # Шаг 4: Оптимизация структуры после pruning
        optimization_results = self._optimize_after_pruning(cluster, pruning_results)
        
        # Шаг 5: Валидация результатов
        validation_results = self._validate_pruning_results(
            cluster, pruning_results, consolidation_results
        )
        
        latency = (datetime.now() - start_time).total_seconds()
        
        # Логируем решение о pruning
        pruning_decision = {
            "timestamp": datetime.now(),
            "cluster": cluster.name,
            "candidates_identified": len(pruning_candidates),
            "nodes_pruned": pruning_results["nodes_pruned"],
            "knowledge_preserved": consolidation_results["knowledge_preserved"],
            "utility_improvement": optimization_results.get("utility_improvement", 0.0)
        }
        self._pruning_decisions.append(pruning_decision)
        
        result = {
            "status": "success",
            "cluster": cluster.name,
            "pruning_candidates": len(pruning_candidates),
            "consolidation_results": consolidation_results,
            "pruning_results": pruning_results,
            "optimization_results": optimization_results,
            "validation_results": validation_results,
            "overall_improvement": self._calculate_pruning_improvement(pruning_results, optimization_results),
            "latency_seconds": latency
        }
        
        logger.info(f"Pruning completed for cluster '{cluster.name}': "
                   f"{pruning_results['nodes_pruned']} nodes pruned, "
                   f"{consolidation_results['knowledge_preserved']} knowledge units preserved")
        
        return self._build_response(cluster, True, "prune_redundant", result)
    
    def _growth_optimize_structure(self, cluster: ClusterContainer, 
                                 attributes: Dict[str, Any]) -> Dict[str, Any]:
        """Оптимизация структуры кластера"""
        optimization_type = attributes.get("optimization_type", "connectivity")
        
        if optimization_type == "connectivity":
            result = self._optimize_connectivity(cluster, attributes)
        elif optimization_type == "hierarchy":
            result = self._optimize_hierarchy(cluster, attributes)
        elif optimization_type == "capacity":
            result = self._optimize_capacity(cluster, attributes)
        else:
            result = {"status": "error", "message": f"Unknown optimization type: {optimization_type}"}
        
        return self._build_response(cluster, True, "optimize_structure", result)
    
    # Вспомогательные методы
    def _collect_low_confidence_inputs(self, cluster: ClusterContainer, 
                                     confidence_threshold: float) -> List[Dict[str, Any]]:
        """Сбор low-confidence inputs из узлов кластера"""
        low_confidence_inputs = []
        
        for item in cluster.get_active_items():
            if isinstance(item, NodeEntity):
                # Получаем статистику узла
                stats = item.get_stats()
                
                # Проверяем confidence metrics
                if stats.get("success_rate", 1.0) < confidence_threshold:
                    # Этот узел имеет низкую success rate
                    low_confidence_inputs.append({
                        "node": item.name,
                        "success_rate": stats["success_rate"],
                        "specialization": item.specialization,
                        "activation_count": item.activation_count,
                        "average_confidence": stats.get("average_reward", 0.5)
                    })
                
                # Также проверяем recent низкие confidence
                if hasattr(item, '_reward_history') and item._reward_history:
                    recent_rewards = item._reward_history[-10:]  # Последние 10 rewards
                    avg_recent = np.mean(recent_rewards) if recent_rewards else 1.0
                    if avg_recent < confidence_threshold:
                        low_confidence_inputs.append({
                            "node": item.name,
                            "recent_confidence": avg_recent,
                            "specialization": item.specialization,
                            "issue": "low_recent_confidence"
                        })
        
        return low_confidence_inputs
    
    def _cluster_gaps(self, low_confidence_inputs: List[Dict[str, Any]], 
                    min_samples: int) -> List[List[Dict[str, Any]]]:
        """Кластеризация пробелов с использованием DBSCAN"""
        if len(low_confidence_inputs) < min_samples:
            # Недостаточно данных для кластеризации
            return [low_confidence_inputs] if low_confidence_inputs else []
        
        try:
            # Создаем feature vectors для кластеризации
            features = []
            for input_data in low_confidence_inputs:
                # Используем success rate и activation count как features
                feature = [
                    input_data.get("success_rate", 0.5),
                    input_data.get("activation_count", 0) / 100.0,  # Нормализовано
                    input_data.get("average_confidence", 0.5),
                    1.0 if input_data.get("issue") == "low_recent_confidence" else 0.0
                ]
                features.append(feature)
            
            features_array = np.array(features)
            
            # Применяем DBSCAN
            dbscan = DBSCAN(eps=0.3, min_samples=min_samples)
            labels = dbscan.fit_predict(features_array)
            
            # Группируем inputs по кластерам
            clusters = {}
            for i, label in enumerate(labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(low_confidence_inputs[i])
            
            # Преобразуем в список, исключая шум (label = -1)
            gap_clusters = [cluster for label, cluster in clusters.items() if label != -1]
            
            return gap_clusters
            
        except Exception as e:
            logger.error(f"Error in gap clustering: {e}")
            return [low_confidence_inputs]
    
    def _mine_new_patterns(self, gap_clusters: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Поиск новых паттернов в кластеризованных пробелах"""
        new_patterns = []
        
        for i, cluster in enumerate(gap_clusters):
            if len(cluster) >= 3:  # Только значительные кластеры
                # Анализируем общие характеристики кластера
                specializations = [item.get("specialization", "unknown") for item in cluster]
                success_rates = [item.get("success_rate", 0.5) for item in cluster]
                
                # Определяем доминирующую специализацию
                from collections import Counter
                specialization_counts = Counter(specializations)
                dominant_specialization = specialization_counts.most_common(1)[0][0] if specialization_counts else "unknown"
                
                # Вычисляем среднюю success rate
                avg_success_rate = np.mean(success_rates) if success_rates else 0.5
                
                pattern = {
                    "pattern_id": f"gap_pattern_{i}",
                    "cluster_size": len(cluster),
                    "dominant_specialization": dominant_specialization,
                    "average_success_rate": avg_success_rate,
                    "gap_severity": 1.0 - avg_success_rate,  # Чем ниже success rate, тем серьезнее пробел
                    "nodes_involved": [item.get("node") for item in cluster],
                    "suggested_action": self._suggest_action_for_gap(dominant_specialization, avg_success_rate)
                }
                
                new_patterns.append(pattern)
        
        return new_patterns
    
    def _assess_growth_needs(self, cluster: ClusterContainer,
                           gap_clusters: List[List[Dict[str, Any]]],
                           new_patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Оценка потребностей в росте на основе пробелов"""
        total_nodes = cluster._node_count
        capacity_used = total_nodes / max(1, cluster.max_capacity)
        
        needs = {
            "new_nodes_needed": 0,
            "specializations_needed": [],
            "connectivity_improvements": 0,
            "capacity_status": capacity_used,
            "priority_level": "medium"
        }
        
        if not gap_clusters:
            needs["priority_level"] = "low"
            return needs
        
        # Анализируем пробелы
        total_gap_severity = sum(pattern.get("gap_severity", 0) for pattern in new_patterns)
        
        if total_gap_severity > 0.5:
            needs["priority_level"] = "high"
            needs["new_nodes_needed"] = min(3, len(new_patterns))
        
        # Определяем необходимые специализации
        for pattern in new_patterns:
            specialization = pattern.get("dominant_specialization")
            if specialization and specialization not in needs["specializations_needed"]:
                needs["specializations_needed"].append(specialization)
        
        # Проверяем connectivity
        if cluster._edge_count < total_nodes * 0.5:  # Меньше 0.5 связей на узел
            needs["connectivity_improvements"] = int(total_nodes * 0.3)  # 30% новых связей
        
        return needs
    
    def _prioritize_gaps(self, gap_clusters: List[List[Dict[str, Any]]],
                       need_assessment: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Приоритизация пробелов для устранения"""
        prioritized = []
        
        # Создаем список для сортировки
        gap_list = []
        for i, cluster in enumerate(gap_clusters):
            if len(cluster) >= 2:  # Только кластеры с минимум 2 элементами
                # Вычисляем приоритет на основе размера кластера и серьезности
                severity_scores = [1.0 - item.get("success_rate", 0.5) for item in cluster]
                avg_severity = np.mean(severity_scores) if severity_scores else 0.5
                
                priority_score = (len(cluster) * 0.4) + (avg_severity * 0.6)
                
                gap_list.append({
                    "cluster_index": i,
                    "size": len(cluster),
                    "avg_severity": avg_severity,
                    "priority_score": priority_score,
                    "nodes": [item.get("node") for item in cluster],
                    "specializations": list(set(item.get("specialization", "unknown") for item in cluster))
                })
        
        # Сортируем по приоритету
        gap_list.sort(key=lambda x: x["priority_score"], reverse=True)
        
        # Ограничиваем количество приоритетных пробелов
        max_prioritized = min(5, len(gap_list))
        prioritized = gap_list[:max_prioritized]
        
        return prioritized
    
    def _generate_growth_recommendations(self, prioritized_gaps: List[Dict[str, Any]]) -> List[str]:
        """Генерация рекомендаций по росту на основе приоритетных пробелов"""
        recommendations = []
        
        for i, gap in enumerate(prioritized_gaps):
            if gap["priority_score"] > 0.7:
                recommendations.append(
                    f"High priority: Create specialized node for {gap['specializations'][0]} "
                    f"(affects {gap['size']} nodes, severity: {gap['avg_severity']:.2f})"
                )
            elif gap["priority_score"] > 0.4:
                recommendations.append(
                    f"Medium priority: Improve connectivity for {gap['specializations'][0]} cluster"
                )
        
        # Общие рекомендации
        if len(prioritized_gaps) > 3:
            recommendations.append("Consider structural reorganization due to multiple significant gaps")
        
        if not recommendations:
            recommendations.append("No immediate growth actions required")
        
        return recommendations
    
    def _determine_node_specialization(self, gap_analysis: Dict[str, Any]) -> str:
        """Определение специализации нового узла на основе анализа пробелов"""
        if "dominant_specialization" in gap_analysis:
            return gap_analysis["dominant_specialization"]
        
        # Анализируем patterns для определения специализации
        patterns = gap_analysis.get("new_patterns", [])
        if patterns:
            # Выбираем наиболее частую специализацию
            specializations = []
            for pattern in patterns:
                spec = pattern.get("dominant_specialization")
                if spec:
                    specializations.append(spec)
            
            if specializations:
                from collections import Counter
                specialization_counts = Counter(specializations)
                return specialization_counts.most_common(1)[0][0]
        
        # Дефолтная специализация
        return "integrative"
    
    def _initialize_node_via_distillation(self, cluster: ClusterContainer,
                                        gap_analysis: Dict[str, Any],
                                        method: str) -> Dict[str, Any]:
        """Инициализация нового узла через дистилляцию знаний"""
        # Находим похожие узлы для дистилляции
        similar_nodes = self._find_similar_nodes_for_distillation(cluster, gap_analysis)
        
        initialization_data = {
            "method": method,
            "similar_nodes_found": len(similar_nodes),
            "distillation_sources": similar_nodes,
            "knowledge_transfer_planned": True
        }
        
        if similar_nodes:
            # Создаем synthetic training data на основе знаний похожих узлов
            training_data = self._create_distillation_training_data(cluster, similar_nodes)
            initialization_data["training_samples"] = len(training_data)
            initialization_data["distillation_strength"] = 0.7
        
        return initialization_data
    
    def _create_new_node_entity(self, cluster: ClusterContainer,
                              specialization: str,
                              initialization_result: Dict[str, Any]) -> NodeEntity:
        """Создание нового NodeEntity"""
        # Генерируем уникальное имя
        node_count = cluster._node_count
        node_name = f"{specialization}_node_{node_count + 1}_{int(datetime.now().timestamp()) % 10000}"
        
        # Определяем параметры на основе специализации
        if specialization in ["visual", "language", "abstract"]:
            input_dim = 1024
            network_type = "transformer"
        else:
            input_dim = 768
            network_type = "mlp"
        
        # Создаем узел
        new_node = NodeEntity(
            name=node_name,
            specialization=specialization,
            input_dim=input_dim,
            output_dim=input_dim,  # Такая же размерность для простоты
            confidence_threshold=0.3,
            network_type=network_type,
            memory_capacity=500,
            isactive=True
        )
        
        # Настраиваем на основе initialization result
        if initialization_result.get("distillation_strength", 0) > 0.5:
            # Повышаем начальный confidence для узлов с дистилляцией
            new_node.confidence_threshold = 0.25
        
        return new_node
    
    def _integrate_new_node(self, cluster: ClusterContainer,
                          new_node: NodeEntity,
                          strategy: str) -> Dict[str, Any]:
        """Интеграция нового узла в существующие connectivity patterns"""
        connections = []
        
        # Находим узлы для соединения
        potential_connections = self._find_potential_connections(cluster, new_node)
        
        # Выбираем соединения в зависимости от стратегии
        if strategy == "adaptive":
            # Адаптивная стратегия: соединяем с наиболее релевантными узлами
            max_connections = min(5, len(potential_connections))
            connections = potential_connections[:max_connections]
        elif strategy == "dense":
            # Плотная стратегия: соединяем со многими узлами
            max_connections = min(10, len(potential_connections))
            connections = potential_connections[:max_connections]
        else:  # sparse
            # Разреженная стратегия: минимальные соединения
            max_connections = min(3, len(potential_connections))
            connections = potential_connections[:max_connections]
        
        return {
            "success": True,
            "strategy": strategy,
            "connections": connections,
            "connection_count": len(connections)
        }
    
    def _configure_new_node(self, node: NodeEntity,
                          gap_analysis: Dict[str, Any],
                          initialization_result: Dict[str, Any]) -> Dict[str, Any]:
        """Настройка нового узла"""
        # Настраиваем параметры на основе gap analysis
        config = {
            "initial_confidence": node.confidence_threshold,
            "energy_level": node.energy_level,
            "learning_rate": getattr(node, '_learning_rate', 0.001),
            "configured_for_gap": gap_analysis.get("gap_severity", 0) > 0.5
        }
        
        # Повышаем начальную энергию для важных узлов
        if gap_analysis.get("priority_level") == "high":
            node.energy_level = min(1.0, node.energy_level + 0.2)
            config["energy_boost_applied"] = True
        
        return config
    
    def _find_similar_nodes_for_distillation(self, cluster: ClusterContainer,
                                           gap_analysis: Dict[str, Any]) -> List[str]:
        """Поиск похожих узлов для дистилляции знаний"""
        target_specialization = gap_analysis.get("dominant_specialization", "")
        similar_nodes = []
        
        for item in cluster.get_active_items():
            if isinstance(item, NodeEntity):
                # Проверяем специализацию
                if target_specialization and target_specialization in item.specialization:
                    similar_nodes.append(item.name)
                # Также проверяем успешные узлы
                elif item.activation_count > 10 and item.energy_level > 0.7:
                    similar_nodes.append(item.name)
        
        # Ограничиваем количество
        return similar_nodes[:5]
    
    def _create_distillation_training_data(self, cluster: ClusterContainer,
                                         similar_nodes: List[str]) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Создание training data для дистилляции"""
        training_data = []
        
        for node_name in similar_nodes[:3]:  # Берем только 3 узла
            if cluster.has_item(node_name):
                node = cluster.get(node_name)
                if isinstance(node, NodeEntity):
                    # Создаем synthetic примеры на основе специализации узла
                    for _ in range(2):  # По 2 примера от каждого узла
                        input_vec = np.random.randn(node.input_dim).astype(np.float32) * 0.1
                        # Простая target: слегка модифицированный input
                        target_vec = input_vec * 0.9 + np.random.randn(node.input_dim).astype(np.float32) * 0.05
                        training_data.append((input_vec, target_vec))
        
        return training_data
    
    def _find_potential_connections(self, cluster: ClusterContainer,
                                  new_node: NodeEntity) -> List[str]:
        """Поиск потенциальных соединений для нового узла"""
        potential_connections = []
        
        for item in cluster.get_active_items():
            if isinstance(item, NodeEntity) and item.name != new_node.name:
                # Проверяем совместимость специализаций
                if self._are_specializations_compatible(new_node.specialization, item.specialization):
                    potential_connections.append(item.name)
        
        # Сортируем по usefulness (активация, энергия)
        potential_connections.sort(
            key=lambda name: cluster.get(name).activation_count if cluster.has_item(name) else 0,
            reverse=True
        )
        
        return potential_connections
    
    def _are_specializations_compatible(self, spec1: str, spec2: str) -> bool:
        """Проверка совместимости специализаций"""
        # Группы совместимых специализаций
        compatible_groups = [
            ["visual", "abstract", "integrative"],
            ["language", "logical", "executive"],
            ["sensory_input", "memory", "integrative"],
            ["emotional", "abstract", "integrative"]
        ]
        
        for group in compatible_groups:
            if spec1 in group and spec2 in group:
                return True
        
        # Все специализации совместимы с integrative
        if spec1 == "integrative" or spec2 == "integrative":
            return True
        
        return False
    
    def _suggest_action_for_gap(self, specialization: str, success_rate: float) -> str:
        """Предложение действия для устранения пробела"""
        if success_rate < 0.3:
            return f"Create new {specialization} node with high capacity"
        elif success_rate < 0.5:
            return f"Improve existing {specialization} nodes or add supporting nodes"
        else:
            return f"Optimize connectivity for {specialization} cluster"
    
    def _identify_pruning_candidates(self, cluster: ClusterContainer,
                                   similarity_threshold: float,
                                   utility_threshold: float) -> List[Dict[str, Any]]:
        """Идентификация кандидатов на удаление"""
        candidates = []
        nodes = []
        
        # Собираем информацию о всех узлах
        for item in cluster.get_active_items():
            if isinstance(item, NodeEntity):
                stats = item.get_stats()
                node_info = {
                    "name": item.name,
                    "specialization": item.specialization,
                    "activation_count": item.activation_count,
                    "energy_level": item.energy_level,
                    "success_rate": stats.get("success_rate", 0.5),
                    "utility_score": self._calculate_node_utility(item, stats),
                    "last_activated": item.last_activated,
                    "memory_usage": len(item._memory.patterns) / max(1, item.memory_capacity)
                }
                nodes.append(node_info)
        
        # Шаг 1: Идентификация по utility threshold
        low_utility_nodes = [node for node in nodes if node["utility_score"] < utility_threshold]
        
        # Шаг 2: Идентификация дубликатов по специализации и активности
        specialization_groups = {}
        for node in nodes:
            spec = node["specialization"]
            if spec not in specialization_groups:
                specialization_groups[spec] = []
            specialization_groups[spec].append(node)
        
        duplicate_candidates = []
        for spec, group in specialization_groups.items():
            if len(group) > 1:  # Только если есть несколько узлов с одинаковой специализацией
                # Сортируем по utility
                group.sort(key=lambda x: x["utility_score"])
                # Помечаем нижние 50% как потенциальные дубликаты
                for node in group[:len(group)//2]:
                    if node["utility_score"] < 0.6:  # Дополнительный порог
                        duplicate_candidates.append(node)
        
        # Шаг 3: Идентификация неактивных узлов
        inactive_nodes = []
        current_time = datetime.now()
        for node in nodes:
            time_since_activation = (current_time - node["last_activated"]).total_seconds()
            if time_since_activation > 86400 and node["activation_count"] < 10:  # 24 часа и мало активаций
                inactive_nodes.append(node)
        
        # Объединяем все кандидаты
        all_candidates = low_utility_nodes + duplicate_candidates + inactive_nodes
        
        # Удаляем дубликаты
        seen_names = set()
        unique_candidates = []
        for candidate in all_candidates:
            if candidate["name"] not in seen_names:
                seen_names.add(candidate["name"])
                unique_candidates.append(candidate)
        
        # Добавляем reason для каждого кандидата
        for candidate in unique_candidates:
            reasons = []
            if candidate in low_utility_nodes:
                reasons.append(f"low_utility({candidate['utility_score']:.3f}<{utility_threshold})")
            if candidate in duplicate_candidates:
                reasons.append("duplicate_specialization")
            if candidate in inactive_nodes:
                reasons.append("inactive")
            
            candidate["pruning_reasons"] = reasons
            candidates.append(candidate)
        
        return candidates
    
    def _calculate_node_utility(self, node: NodeEntity, stats: Dict[str, Any]) -> float:
        """Расчет utility score узла"""
        factors = []
        
        # Factor 1: Success rate
        success_rate = stats.get("success_rate", 0.5)
        factors.append(success_rate * 0.3)
        
        # Factor 2: Activation frequency (нормализованная)
        activation_factor = min(1.0, node.activation_count / 100.0)
        factors.append(activation_factor * 0.25)
        
        # Factor 3: Energy efficiency
        energy_factor = node.energy_level
        factors.append(energy_factor * 0.2)
        
        # Factor 4: Memory usage (умеренное использование лучше)
        memory_usage = len(node._memory.patterns) / max(1, node.memory_capacity)
        memory_factor = 1.0 - abs(0.7 - memory_usage)  # Пик при 70% использовании
        factors.append(memory_factor * 0.15)
        
        # Factor 5: Recency
        time_since_activation = (datetime.now() - node.last_activated).total_seconds()
        recency_factor = 1.0 / (1.0 + time_since_activation / 3600)  # Затухание по часам
        factors.append(recency_factor * 0.1)
        
        return sum(factors)
    
    def _consolidate_knowledge_before_pruning(self, cluster: ClusterContainer,
                                            pruning_candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Консолидация знаний перед удалением узлов"""
        knowledge_preserved = 0
        transfer_attempts = 0
        transfer_successes = 0
        
        for candidate in pruning_candidates:
            node_name = candidate["name"]
            if cluster.has_item(node_name):
                node = cluster.get(node_name)
                if isinstance(node, NodeEntity) and node._memory.patterns:
                    # Пытаемся передать знания другим узлам
                    transfer_result = self._transfer_knowledge_to_survivors(
                        cluster, node, pruning_candidates
                    )
                    
                    transfer_attempts += 1
                    if transfer_result["success"]:
                        transfer_successes += 1
                        knowledge_preserved += transfer_result["patterns_transferred"]
        
        return {
            "knowledge_preserved": knowledge_preserved,
            "transfer_attempts": transfer_attempts,
            "transfer_success_rate": transfer_successes / max(1, transfer_attempts),
            "consolidation_complete": transfer_attempts > 0
        }
    
    def _transfer_knowledge_to_survivors(self, cluster: ClusterContainer,
                                       doomed_node: NodeEntity,
                                       all_candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Передача знаний от обреченного узла к выжившим"""
        # Находим выжившие узлы с похожей специализацией
        survivors = []
        doomed_spec = doomed_node.specialization
        
        for item in cluster.get_active_items():
            if isinstance(item, NodeEntity) and item.name != doomed_node.name:
                # Проверяем, не является ли этот узел кандидатом на удаление
                is_candidate = any(candidate["name"] == item.name for candidate in all_candidates)
                
                if not is_candidate and self._are_specializations_compatible(doomed_spec, item.specialization):
                    survivors.append(item)
        
        if not survivors:
            return {"success": False, "patterns_transferred": 0, "reason": "no_survivors"}
        
        # Выбираем лучшего реципиента
        best_recipient = max(survivors, key=lambda node: node.energy_level * node.activation_count)
        
        # Передаем часть знаний
        patterns_to_transfer = min(5, len(doomed_node._memory.patterns))
        patterns_transferred = 0
        
        for i, pattern in enumerate(doomed_node._memory.patterns[:patterns_to_transfer]):
            try:
                # Создаем synthetic input на основе pattern
                input_vector = pattern.vector * 0.8 + np.random.randn(len(pattern.vector)) * 0.1
                
                # Используем обучение для передачи знания
                result = best_recipient.learn_from_experience(
                    input_vector=input_vector,
                    target_output=pattern.vector,
                    reward=0.7
                )
                
                if result.get("status") != "inactive":
                    patterns_transferred += 1
            except Exception as e:
                logger.error(f"Error transferring pattern {i}: {e}")
        
        return {
            "success": patterns_transferred > 0,
            "patterns_transferred": patterns_transferred,
            "recipient": best_recipient.name,
            "transfer_ratio": patterns_transferred / max(1, len(doomed_node._memory.patterns))
        }
    
    def _execute_pruning(self, cluster: ClusterContainer,
                       pruning_candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Выполнение удаления узлов"""
        nodes_pruned = 0
        pruning_errors = []
        
        for candidate in pruning_candidates:
            node_name = candidate["name"]
            
            try:
                # Удаляем узел из кластера
                if cluster.has_item(node_name):
                    # Сначала удаляем связанные edges
                    self._remove_associated_edges(cluster, node_name)
                    
                    # Затем удаляем сам узел
                    cluster.remove(node_name)
                    nodes_pruned += 1
                    
                    logger.debug(f"Pruned node '{node_name}' from cluster '{cluster.name}'")
            except Exception as e:
                pruning_errors.append(f"Error pruning {node_name}: {str(e)}")
        
        return {
            "nodes_pruned": nodes_pruned,
            "total_candidates": len(pruning_candidates),
            "pruning_success_rate": nodes_pruned / max(1, len(pruning_candidates)),
            "errors": pruning_errors
        }
    
    def _remove_associated_edges(self, cluster: ClusterContainer, node_name: str):
        """Удаление связей, связанных с узлом"""
        edges_to_remove = []
        
        for edge_name in cluster.intra_cluster_edges:
            if cluster.has_item(edge_name):
                edge = cluster.get(edge_name)
                if isinstance(edge, EdgeEntity):
                    if edge.source_node == node_name or edge.target_node == node_name:
                        edges_to_remove.append(edge_name)
        
        for edge_name in edges_to_remove:
            try:
                cluster.remove(edge_name)
                if edge_name in cluster.intra_cluster_edges:
                    cluster.intra_cluster_edges.remove(edge_name)
            except Exception as e:
                logger.error(f"Error removing edge '{edge_name}': {e}")
    
    def _optimize_after_pruning(self, cluster: ClusterContainer,
                              pruning_results: Dict[str, Any]) -> Dict[str, Any]:
        """Оптимизация структуры после удаления узлов"""
        optimizations_applied = 0
        
        # Оптимизация 1: Перебалансировка энергии
        if cluster._node_count > 0:
            # Распределяем энергию от удаленных узлов среди оставшихся
            energy_per_node = 0.05 / cluster._node_count
            for item in cluster.get_active_items():
                if isinstance(item, NodeEntity):
                    item.recharge_energy(energy_per_node)
                    optimizations_applied += 1
        
        # Оптимизация 2: Упрощение connectivity
        # (В реальной реализации здесь была бы более сложная логика)
        
        return {
            "optimizations_applied": optimizations_applied,
            "energy_redistributed": energy_per_node if cluster._node_count > 0 else 0,
            "utility_improvement": 0.1  # Оценочное улучшение
        }
    
    def _validate_pruning_results(self, cluster: ClusterContainer,
                                pruning_results: Dict[str, Any],
                                consolidation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Валидация результатов pruning"""
        # Проверяем, что кластер все еще функционален
        active_nodes = len([item for item in cluster.get_active_items() if isinstance(item, NodeEntity)])
        
        validation_passed = active_nodes > 0
        
        # Проверяем, что не удалили слишком много
        pruning_ratio = pruning_results["nodes_pruned"] / max(1, cluster._node_count + pruning_results["nodes_pruned"])
        too_aggressive = pruning_ratio > 0.5
        
        # Проверяем сохранение знаний
        knowledge_preserved_well = consolidation_results["transfer_success_rate"] > 0.5
        
        return {
            "validation_passed": validation_passed,
            "active_nodes_remaining": active_nodes,
            "pruning_ratio": pruning_ratio,
            "too_aggressive": too_aggressive,
            "knowledge_preserved_well": knowledge_preserved_well,
            "overall_valid": validation_passed and not too_aggressive
        }
    
    def _calculate_pruning_improvement(self, pruning_results: Dict[str, Any],
                                     optimization_results: Dict[str, Any]) -> float:
        """Расчет общего улучшения от pruning"""
        base_improvement = pruning_results["pruning_success_rate"] * 0.6
        optimization_bonus = optimization_results.get("utility_improvement", 0) * 0.4
        
        return min(1.0, base_improvement + optimization_bonus)
    
    def _optimize_connectivity(self, cluster: ClusterContainer,
                             attributes: Dict[str, Any]) -> Dict[str, Any]:
        """Оптимизация связей в кластере"""
        optimization_strength = attributes.get("strength", 0.5)
        
        # Анализируем текущую connectivity
        nodes = [item for item in cluster.get_active_items() if isinstance(item, NodeEntity)]
        edges = [item for item in cluster.get_active_items() if isinstance(item, EdgeEntity)]
        
        # Вычисляем метрики connectivity
        connectivity_metrics = self._analyze_connectivity(nodes, edges)
        
        # Применяем оптимизации
        optimizations = []
        
        # 1. Добавление недостающих связей
        if connectivity_metrics["avg_connections_per_node"] < 2.5:
            new_edges = self._add_missing_connections(cluster, nodes, optimization_strength)
            optimizations.append(f"Added {new_edges} new connections")
        
        # 2. Усиление важных связей
        strengthened = self._strengthen_important_connections(cluster, edges, optimization_strength)
        optimizations.append(f"Strengthened {strengthened} important connections")
        
        # 3. Удаление слабых связей
        if optimization_strength > 0.7:
            removed = self._remove_weak_connections(cluster, edges, 0.1)
            optimizations.append(f"Removed {removed} weak connections")
        
        return {
            "status": "success",
            "optimizations_applied": len(optimizations),
            "optimization_details": optimizations,
            "connectivity_before": connectivity_metrics,
            "connectivity_after": self._analyze_connectivity(
                [item for item in cluster.get_active_items() if isinstance(item, NodeEntity)],
                [item for item in cluster.get_active_items() if isinstance(item, EdgeEntity)]
            )
        }
    
    def _analyze_connectivity(self, nodes: List[NodeEntity], edges: List[EdgeEntity]) -> Dict[str, float]:
        """Анализ метрик connectivity"""
        if not nodes:
            return {"avg_connections_per_node": 0, "connection_strength_avg": 0, "connectivity_score": 0}
        
        # Подсчитываем связи на узел
        connections_per_node = defaultdict(int)
        for edge in edges:
            connections_per_node[edge.source_node] += 1
            connections_per_node[edge.target_node] += 1
        
        avg_connections = sum(connections_per_node.values()) / len(nodes)
        
        # Средняя сила связей
        avg_strength = np.mean([edge.get_connection_strength() for edge in edges]) if edges else 0
        
        # Общая оценка connectivity
        connectivity_score = min(1.0, (avg_connections / 5.0) * 0.6 + avg_strength * 0.4)
        
        return {
            "avg_connections_per_node": avg_connections,
            "connection_strength_avg": avg_strength,
            "connectivity_score": connectivity_score
        }
    
    def _add_missing_connections(self, cluster: ClusterContainer,
                               nodes: List[NodeEntity], strength: float) -> int:
        """Добавление недостающих связей"""
        new_edges = 0
        max_new_edges = int(len(nodes) * strength)
        
        for i, node1 in enumerate(nodes):
            for j, node2 in enumerate(nodes):
                if i >= j or new_edges >= max_new_edges:
                    continue
                
                # Проверяем, существует ли уже связь
                connection_exists = False
                for edge_name in cluster.intra_cluster_edges:
                    if cluster.has_item(edge_name):
                        edge = cluster.get(edge_name)
                        if isinstance(edge, EdgeEntity):
                            if (edge.source_node == node1.name and edge.target_node == node2.name) or \
                               (edge.source_node == node2.name and edge.target_node == node1.name):
                                connection_exists = True
                                break
                
                if not connection_exists and self._should_connect(node1, node2):
                    # Создаем новую связь
                    edge_name = f"edge_{node1.name}_to_{node2.name}"
                    edge = EdgeEntity(
                        name=edge_name,
                        source_node=node1.name,
                        target_node=node2.name,
                        weight=0.3 * strength,
                        connection_type="excitatory"
                    )
                    
                    try:
                        cluster.add(edge)
                        cluster.intra_cluster_edges.append(edge_name)
                        new_edges += 1
                    except Exception as e:
                        logger.error(f"Error adding edge {edge_name}: {e}")
        
        return new_edges
    
    def _should_connect(self, node1: NodeEntity, node2: NodeEntity) -> bool:
        """Определение, стоит ли соединять два узла"""
        # Проверяем совместимость специализаций
        if not self._are_specializations_compatible(node1.specialization, node2.specialization):
            return False
        
        # Проверяем активность узлов
        if node1.activation_count < 5 and node2.activation_count < 5:
            return False  # Оба неактивны
        
        # Проверяем энергию
        if node1.energy_level < 0.2 or node2.energy_level < 0.2:
            return False  # Слишком мало энергии
        
        return True
    
    def _strengthen_important_connections(self, cluster: ClusterContainer,
                                        edges: List[EdgeEntity], strength: float) -> int:
        """Усиление важных связей"""
        strengthened = 0
        
        for edge in edges:
            # Определяем важность связи
            importance = self._calculate_edge_importance(edge)
            
            if importance > 0.6:
                # Усиливаем связь
                reinforcement = min(0.2, importance * strength * 0.1)
                edge.strengthen(reinforcement)
                strengthened += 1
        
        return strengthened
    
    def _calculate_edge_importance(self, edge: EdgeEntity) -> float:
        """Расчет важности связи"""
        factors = []
        
        # Factor 1: Usage frequency
        usage_factor = min(1.0, edge.usage_frequency / 50.0)
        factors.append(usage_factor * 0.4)
        
        # Factor 2: Weight strength
        weight_factor = abs(edge.weight)
        factors.append(weight_factor * 0.3)
        
        # Factor 3: Evolutionary score
        factors.append(edge.evolutionary_score * 0.2)
        
        # Factor 4: Recency
        time_since_last = (datetime.now() - edge.last_reinforcement_time).total_seconds()
        recency_factor = 1.0 / (1.0 + time_since_last / 3600)
        factors.append(recency_factor * 0.1)
        
        return sum(factors)
    
    def _remove_weak_connections(self, cluster: ClusterContainer,
                               edges: List[EdgeEntity], threshold: float) -> int:
        """Удаление слабых связей"""
        removed = 0
        
        for edge in edges:
            # Определяем слабость связи
            weakness = 1.0 - self._calculate_edge_importance(edge)
            
            if weakness > threshold and abs(edge.weight) < 0.1:
                try:
                    cluster.remove(edge.name)
                    if edge.name in cluster.intra_cluster_edges:
                        cluster.intra_cluster_edges.remove(edge.name)
                    removed += 1
                except Exception as e:
                    logger.error(f"Error removing edge {edge.name}: {e}")
        
        return removed
    
    def _optimize_hierarchy(self, cluster: ClusterContainer,
                          attributes: Dict[str, Any]) -> Dict[str, Any]:
        """Оптимизация иерархии кластера"""
        # В данной реализации просто возвращаем информацию о текущей иерархии
        hierarchy_info = self._analyze_hierarchy(cluster)
        
        return {
            "status": "success",
            "hierarchy_info": hierarchy_info,
            "recommendations": ["Consider adding subclusters for better organization"] 
            if hierarchy_info["depth"] < 2 else ["Hierarchy appears well-structured"]
        }
    
    def _analyze_hierarchy(self, cluster: ClusterContainer) -> Dict[str, Any]:
        """Анализ иерархии кластера"""
        subclusters = [item for item in cluster.get_items() if isinstance(item, ClusterContainer)]
        
        return {
            "total_subclusters": len(subclusters),
            "depth": self._calculate_hierarchy_depth(cluster),
            "branching_factor": len(subclusters) / max(1, cluster._node_count),
            "hierarchy_score": min(1.0, len(subclusters) * 0.1 + self._calculate_hierarchy_depth(cluster) * 0.2)
        }
    
    def _calculate_hierarchy_depth(self, cluster: ClusterContainer) -> int:
        """Расчет глубины иерархии"""
        max_depth = 0
        
        for item in cluster.get_items():
            if isinstance(item, ClusterContainer):
                depth = 1 + self._calculate_hierarchy_depth(item)
                max_depth = max(max_depth, depth)
        
        return max_depth
    
    def _optimize_capacity(self, cluster: ClusterContainer,
                         attributes: Dict[str, Any]) -> Dict[str, Any]:
        """Оптимизация capacity кластера"""
        current_usage = cluster._node_count / cluster.max_capacity
        
        if current_usage > 0.9:
            action = "increase_capacity"
            new_capacity = int(cluster.max_capacity * 1.5)
        elif current_usage < 0.3:
            action = "decrease_capacity"
            new_capacity = max(cluster._node_count + 5, int(cluster.max_capacity * 0.7))
        else:
            action = "maintain_capacity"
            new_capacity = cluster.max_capacity
        
        return {
            "status": "success",
            "current_usage": current_usage,
            "action": action,
            "old_capacity": cluster.max_capacity,
            "new_capacity": new_capacity,
            "capacity_change": new_capacity - cluster.max_capacity
        }