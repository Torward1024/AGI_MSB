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

class MemorySuper(Super):
    """
    Распределенная память в графовой архитектуре AGI.
    
    Реализует механизмы работы с памятью:
    - Мультимодальное хранение опыта
    - Content-addressable поиск
    - Консолидация знаний во время "сна"
    """
    
    OPERATION = "memory"
    
    def __init__(self, manipulator=None, methods=None, cache_size=2048):
        super().__init__(manipulator, methods, cache_size)
        self._experience_buffer = deque(maxlen=1000)
        self._retrieval_patterns = {}
        self._consolidation_schedule = []
    
    def _memory_cluster(self, cluster: ClusterContainer, 
                       attributes: Dict[str, Any]) -> Dict[str, Any]:
        """Операции с памятью в кластере"""
        memory_operation = attributes.get("operation", "store_experience")
        
        if memory_operation == "store_experience":
            return self._memory_store_experience(cluster, attributes)
        elif memory_operation == "retrieve_similar":
            return self._memory_retrieve_similar(cluster, attributes)
        elif memory_operation == "consolidate_sleep":
            return self._memory_consolidate_sleep(cluster, attributes)
        elif memory_operation == "analyze_memory":
            return self._memory_analyze(cluster, attributes)
        else:
            return self._build_response(cluster, False, None, None, 
                                      f"Unknown memory operation: {memory_operation}")
    
    def _memory_store_experience(self, cluster: ClusterContainer, 
                               attributes: Dict[str, Any]) -> Dict[str, Any]:
        """
        Мультимодальное хранение опыта.
        
        Кодирование с векторами и метаданными,
        importance-based сохранение,
        индексация для эффективного поиска.
        """
        if not cluster.isactive:
            return self._build_response(cluster, False, None, None, "Cluster is inactive")
        
        experience_data = attributes.get("experience_data", {})
        importance_score = attributes.get("importance", 0.5)
        encoding_method = attributes.get("encoding_method", "auto")
        
        if not experience_data:
            return self._build_response(cluster, False, None, None, 
                                      "No experience data provided")
        
        start_time = datetime.now()
        
        # Шаг 1: Мультимодальное кодирование
        encoded_experience = self._encode_multimodal_experience(
            experience_data, encoding_method
        )
        
        # Шаг 2: Importance-based фильтрация
        if not self._passes_importance_filter(encoded_experience, importance_score):
            return self._build_response(cluster, True, "store_experience", {
                "status": "filtered_out",
                "cluster": cluster.name,
                "reason": "Importance score too low",
                "importance_score": importance_score
            })
        
        # Шаг 3: Распределение по узлам
        storage_results = self._distribute_to_nodes(
            cluster, encoded_experience, importance_score
        )
        
        # Шаг 4: Индексация для поиска
        indexing_result = self._index_experience(
            cluster, encoded_experience, storage_results
        )
        
        # Шаг 5: Обновление метаданных памяти
        metadata_update = self._update_memory_metadata(
            cluster, encoded_experience, storage_results
        )
        
        latency = (datetime.now() - start_time).total_seconds()
        
        # Сохраняем в буфер опыта
        experience_record = {
            "timestamp": datetime.now(),
            "cluster": cluster.name,
            "experience_id": encoded_experience.get("experience_id", "unknown"),
            "importance": importance_score,
            "nodes_used": storage_results["nodes_used"],
            "storage_success": storage_results["success_count"] > 0
        }
        self._experience_buffer.append(experience_record)
        
        result = {
            "status": "success",
            "cluster": cluster.name,
            "experience_id": encoded_experience.get("experience_id", "unknown"),
            "encoding_method": encoding_method,
            "importance_score": importance_score,
            "storage_results": storage_results,
            "indexing_result": indexing_result,
            "metadata_updated": metadata_update["updated"],
            "total_stored": storage_results["success_count"],
            "storage_time_seconds": latency
        }
        
        logger.info(f"Experience stored in cluster '{cluster.name}': "
                   f"id={encoded_experience.get('experience_id', 'unknown')}, "
                   f"importance={importance_score:.3f}, "
                   f"{storage_results['success_count']} nodes used")
        
        return self._build_response(cluster, True, "store_experience", result)
    
    def _memory_retrieve_similar(self, cluster: ClusterContainer, 
                               attributes: Dict[str, Any]) -> Dict[str, Any]:
        """
        Content-addressable поиск похожих воспоминаний.
        
        Поиск по всему кластеру,
        context-aware сравнение,
        временное взвешивание релевантности.
        """
        if not cluster.isactive:
            return self._build_response(cluster, False, None, None, "Cluster is inactive")
        
        query_vector = attributes.get("query_vector")
        context = attributes.get("context", {})
        similarity_threshold = attributes.get("similarity_threshold", 0.7)
        max_results = attributes.get("max_results", 10)
        
        if query_vector is None:
            return self._build_response(cluster, False, None, None, 
                                      "No query vector provided")
        
        start_time = datetime.now()
        
        # Шаг 1: Content-addressable поиск по узлам
        node_results = self._search_across_nodes(
            cluster, query_vector, similarity_threshold, max_results
        )
        
        # Шаг 2: Context-aware фильтрация
        context_filtered = self._apply_context_filter(
            node_results, context, similarity_threshold
        )
        
        # Шаг 3: Временное взвешивание релевантности
        temporally_weighted = self._apply_temporal_weighting(context_filtered)
        
        # Шаг 4: Ранжирование и выборка результатов
        final_results = self._rank_and_select_results(
            temporally_weighted, max_results
        )
        
        # Шаг 5: Анализ паттернов поиска
        pattern_analysis = self._analyze_retrieval_patterns(
            query_vector, final_results, context
        )
        
        latency = (datetime.now() - start_time).total_seconds()
        
        # Обновляем статистику поиска
        retrieval_key = hash(str(query_vector.tobytes() if hasattr(query_vector, 'tobytes') else str(query_vector)))
        if retrieval_key not in self._retrieval_patterns:
            self._retrieval_patterns[retrieval_key] = {
                "count": 0,
                "first_seen": datetime.now(),
                "last_seen": datetime.now(),
                "average_similarity": 0.0
            }
        
        pattern = self._retrieval_patterns[retrieval_key]
        pattern["count"] += 1
        pattern["last_seen"] = datetime.now()
        
        if final_results:
            pattern["average_similarity"] = np.mean([
                r.get("similarity", 0) for r in final_results
            ])
        
        result = {
            "status": "success",
            "cluster": cluster.name,
            "query_dimensions": query_vector.shape if hasattr(query_vector, 'shape') else "unknown",
            "nodes_searched": node_results["nodes_searched"],
            "total_matches_found": node_results["total_matches"],
            "context_filtered": len(context_filtered),
            "final_results_count": len(final_results),
            "results": final_results,
            "pattern_analysis": pattern_analysis,
            "average_similarity": np.mean([r.get("similarity", 0) for r in final_results]) if final_results else 0.0,
            "retrieval_time_seconds": latency
        }
        
        logger.info(f"Memory retrieval in cluster '{cluster.name}': "
                   f"{len(final_results)} results, "
                   f"avg similarity={result['average_similarity']:.3f}, "
                   f"searched {node_results['nodes_searched']} nodes")
        
        return self._build_response(cluster, True, "retrieve_similar", result)
    
    def _memory_consolidate_sleep(self, cluster: ClusterContainer, 
                                attributes: Dict[str, Any]) -> Dict[str, Any]:
        """
        Консолидация памяти во время "сна".
        
        Оффлайн replay и реорганизация,
        сжатие знаний через autoencoder,
        генерация dream-like паттернов для творчества.
        """
        if not cluster.isactive:
            return self._build_response(cluster, False, None, None, "Cluster is inactive")
        
        consolidation_mode = attributes.get("mode", "standard")
        compression_strength = attributes.get("compression_strength", 0.5)
        creativity_level = attributes.get("creativity_level", 0.3)
        
        start_time = datetime.now()
        
        # Шаг 1: Оффлайн replay важных воспоминаний
        replay_results = self._offline_replay(cluster, consolidation_mode)
        
        # Шаг 2: Реорганизация структуры памяти
        reorganization_results = self._reorganize_memory_structure(
            cluster, replay_results
        )
        
        # Шаг 3: Сжатие знаний через упрощенный autoencoder
        compression_results = self._compress_knowledge(
            cluster, compression_strength
        )
        
        # Шаг 4: Генерация dream-like паттернов
        dream_patterns = []
        if creativity_level > 0.1:
            dream_patterns = self._generate_dream_patterns(
                cluster, creativity_level, replay_results
            )
        
        # Шаг 5: Очистка и оптимизация
        cleanup_results = self._cleanup_memory(cluster, reorganization_results)
        
        # Шаг 6: Обновление консолидационного расписания
        schedule_update = self._update_consolidation_schedule(
            cluster, replay_results, compression_results
        )
        
        latency = (datetime.now() - start_time).total_seconds()
        
        # Записываем сессию консолидации
        consolidation_session = {
            "timestamp": datetime.now(),
            "cluster": cluster.name,
            "mode": consolidation_mode,
            "replay_count": replay_results.get("experiences_replayed", 0),
            "compression_ratio": compression_results.get("compression_ratio", 1.0),
            "dream_patterns_generated": len(dream_patterns),
            "memory_freed_mb": cleanup_results.get("memory_freed_mb", 0),
            "duration_seconds": latency
        }
        self._consolidation_schedule.append(consolidation_session)
        
        result = {
            "status": "success",
            "cluster": cluster.name,
            "consolidation_mode": consolidation_mode,
            "replay_results": replay_results,
            "reorganization_results": reorganization_results,
            "compression_results": compression_results,
            "dream_patterns_generated": len(dream_patterns),
            "dream_patterns_sample": dream_patterns[:3] if dream_patterns else [],
            "cleanup_results": cleanup_results,
            "schedule_updated": schedule_update["updated"],
            "next_recommended_consolidation": schedule_update.get("next_recommended", "24h"),
            "total_processing_time_seconds": latency
        }
        
        logger.info(f"Memory consolidation completed for cluster '{cluster.name}': "
                   f"mode={consolidation_mode}, "
                   f"{replay_results.get('experiences_replayed', 0)} experiences replayed, "
                   f"compression={compression_results.get('compression_ratio', 1.0):.2f}x")
        
        return self._build_response(cluster, True, "consolidate_sleep", result)
    
    def _memory_analyze(self, cluster: ClusterContainer, 
                      attributes: Dict[str, Any]) -> Dict[str, Any]:
        """Анализ состояния памяти кластера"""
        analysis_type = attributes.get("analysis_type", "comprehensive")
        
        if analysis_type == "comprehensive":
            result = self._analyze_memory_comprehensive(cluster)
        elif analysis_type == "performance":
            result = self._analyze_memory_performance(cluster)
        elif analysis_type == "content":
            result = self._analyze_memory_content(cluster)
        else:
            result = {"status": "error", "message": f"Unknown analysis type: {analysis_type}"}
        
        return self._build_response(cluster, True, "analyze_memory", result)
    
    # Вспомогательные методы для _memory_store_experience
    def _encode_multimodal_experience(self, experience_data: Dict[str, Any],
                                    encoding_method: str) -> Dict[str, Any]:
        """Мультимодальное кодирование опыта"""
        encoded = {
            "experience_id": f"exp_{int(datetime.now().timestamp())}_{hash(str(experience_data)) % 10000}",
            "timestamp": datetime.now(),
            "raw_data": experience_data,
            "encodings": {},
            "metadata": {}
        }
        
        # Векторное кодирование
        if "vector_data" in experience_data:
            encoded["encodings"]["vector"] = self._encode_vector(
                experience_data["vector_data"], encoding_method
            )
        
        # Текстовое кодирование
        if "text_data" in experience_data:
            encoded["encodings"]["text"] = self._encode_text(
                experience_data["text_data"]
            )
        
        # Метаданные
        encoded["metadata"] = {
            "source": experience_data.get("source", "unknown"),
            "importance_estimated": experience_data.get("importance", 0.5),
            "modality_count": len(encoded["encodings"]),
            "encoding_method": encoding_method
        }
        
        return encoded
    
    def _encode_vector(self, vector_data: Any, method: str) -> np.ndarray:
        """Кодирование векторных данных"""
        if isinstance(vector_data, np.ndarray):
            return vector_data.copy()
        elif isinstance(vector_data, list):
            return np.array(vector_data, dtype=np.float32)
        else:
            # Дефолтное кодирование
            return np.random.randn(768).astype(np.float32) * 0.1
    
    def _encode_text(self, text_data: str) -> Dict[str, Any]:
        """Кодирование текстовых данных (упрощенное)"""
        # В реальной реализации здесь было бы использование SentenceTransformer или подобного
        words = text_data.lower().split()
        word_count = len(words)
        unique_words = len(set(words))
        
        return {
            "text_preview": text_data[:100] + "..." if len(text_data) > 100 else text_data,
            "word_count": word_count,
            "unique_words": unique_words,
            "vector_placeholder": np.random.randn(768).astype(np.float32) * 0.1
        }
    
    def _passes_importance_filter(self, encoded_experience: Dict[str, Any],
                                importance_threshold: float) -> bool:
        """Фильтрация на основе важности"""
        # Проверяем явную важность
        if encoded_experience["metadata"]["importance_estimated"] >= importance_threshold:
            return True
        
        # Дополнительные критерии важности
        importance_factors = []
        
        # Фактор 1: Разнообразие модальностей
        modality_factor = encoded_experience["metadata"]["modality_count"] / 3.0
        importance_factors.append(modality_factor * 0.3)
        
        # Фактор 2: Размер данных
        if "vector" in encoded_experience["encodings"]:
            vector_norm = np.linalg.norm(encoded_experience["encodings"]["vector"])
            size_factor = min(1.0, vector_norm / 10.0)
            importance_factors.append(size_factor * 0.4)
        
        # Фактор 3: Источник
        source = encoded_experience["metadata"]["source"]
        source_factor = 0.7 if source in ["learning", "important_event"] else 0.3
        importance_factors.append(source_factor * 0.3)
        
        total_importance = sum(importance_factors)
        
        return total_importance >= importance_threshold
    
    def _distribute_to_nodes(self, cluster: ClusterContainer,
                           encoded_experience: Dict[str, Any],
                           importance_score: float) -> Dict[str, Any]:
        """Распределение опыта по узлам кластера"""
        storage_results = {
            "nodes_considered": 0,
            "nodes_used": [],
            "success_count": 0,
            "failure_count": 0,
            "storage_details": []
        }
        
        # Выбираем узлы для хранения на основе специализации
        candidate_nodes = self._select_storage_nodes(cluster, encoded_experience)
        
        for node_name in candidate_nodes:
            if cluster.has_item(node_name):
                node = cluster.get(node_name)
                if isinstance(node, NodeEntity) and node.isactive:
                    storage_results["nodes_considered"] += 1
                    
                    try:
                        # Преобразуем опыт в формат для хранения в узле
                        storage_vector = self._prepare_for_node_storage(
                            encoded_experience, node
                        )
                        
                        # Сохраняем в память узла
                        pattern_id = node.add_to_memory(
                            vector=storage_vector,
                            label=f"exp_{encoded_experience['experience_id']}",
                            confidence=importance_score,
                            metadata={
                                "experience_id": encoded_experience["experience_id"],
                                "timestamp": encoded_experience["timestamp"],
                                "importance": importance_score,
                                "storage_node": node_name
                            }
                        )
                        
                        if pattern_id:
                            storage_results["success_count"] += 1
                            storage_results["nodes_used"].append(node_name)
                            storage_results["storage_details"].append({
                                "node": node_name,
                                "pattern_id": pattern_id,
                                "success": True
                            })
                        else:
                            storage_results["failure_count"] += 1
                            
                    except Exception as e:
                        logger.error(f"Error storing experience in node '{node_name}': {e}")
                        storage_results["failure_count"] += 1
                        storage_results["storage_details"].append({
                            "node": node_name,
                            "error": str(e),
                            "success": False
                        })
        
        return storage_results
    
    def _select_storage_nodes(self, cluster: ClusterContainer,
                            encoded_experience: Dict[str, Any]) -> List[str]:
        """Выбор узлов для хранения опыта"""
        candidate_nodes = []
        
        # Анализируем опыт для определения подходящих специализаций
        experience_type = self._classify_experience(encoded_experience)
        
        for item in cluster.get_active_items():
            if isinstance(item, NodeEntity):
                # Проверяем совместимость специализаций
                if self._is_storage_compatible(item.specialization, experience_type):
                    # Проверяем capacity узла
                    memory_usage = len(item._memory.patterns) / max(1, item.memory_capacity)
                    if memory_usage < 0.8:  # Узел не переполнен
                        candidate_nodes.append(item.name)
        
        # Ограничиваем количество узлов
        max_nodes = min(5, len(candidate_nodes))
        return candidate_nodes[:max_nodes]
    
    def _classify_experience(self, encoded_experience: Dict[str, Any]) -> str:
        """Классификация типа опыта"""
        metadata = encoded_experience["metadata"]
        
        if "text" in encoded_experience["encodings"]:
            return "language"
        elif "vector" in encoded_experience["encodings"]:
            vector = encoded_experience["encodings"]["vector"]
            if len(vector) >= 1024:
                return "visual" if random.random() > 0.5 else "abstract"
            else:
                return "general"
        
        return "general"
    
    def _is_storage_compatible(self, node_specialization: str, 
                             experience_type: str) -> bool:
        """Проверка совместимости специализации узла и типа опыта"""
        compatibility_map = {
            "language": ["language", "integrative", "general"],
            "visual": ["visual", "integrative", "abstract"],
            "abstract": ["abstract", "integrative", "logical"],
            "general": ["general", "integrative", "memory"]
        }
        
        compatible_types = compatibility_map.get(experience_type, ["general", "integrative"])
        return node_specialization in compatible_types
    
    def _prepare_for_node_storage(self, encoded_experience: Dict[str, Any],
                                node: NodeEntity) -> np.ndarray:
        """Подготовка данных для хранения в узле"""
        # Используем векторное представление, если доступно
        if "vector" in encoded_experience["encodings"]:
            vector = encoded_experience["encodings"]["vector"]
        else:
            # Создаем synthetic vector на основе других данных
            vector = np.random.randn(768).astype(np.float32) * 0.1
        
        # Адаптируем размерность к узлу
        target_dim = node.input_dim
        if len(vector) != target_dim:
            # Простая адаптация размерности
            if len(vector) > target_dim:
                vector = vector[:target_dim]
            else:
                # Дополняем zeros
                padded = np.zeros(target_dim, dtype=np.float32)
                padded[:len(vector)] = vector
                vector = padded
        
        return vector
    
    def _index_experience(self, cluster: ClusterContainer,
                        encoded_experience: Dict[str, Any],
                        storage_results: Dict[str, Any]) -> Dict[str, Any]:
        """Индексация опыта для поиска"""
        # В реальной реализации здесь было бы обновление индексов поиска
        # В данной реализации просто регистрируем индексацию
        
        index_entry = {
            "experience_id": encoded_experience["experience_id"],
            "timestamp": encoded_experience["timestamp"],
            "storage_nodes": storage_results["nodes_used"],
            "importance": encoded_experience["metadata"]["importance_estimated"],
            "modalities": list(encoded_experience["encodings"].keys())
        }
        
        return {
            "indexed": True,
            "index_entry": index_entry,
            "searchable": len(storage_results["nodes_used"]) > 0
        }
    
    def _update_memory_metadata(self, cluster: ClusterContainer,
                              encoded_experience: Dict[str, Any],
                              storage_results: Dict[str, Any]) -> Dict[str, Any]:
        """Обновление метаданных памяти кластера"""
        # Обновляем метрики использования памяти
        old_memory_usage = getattr(cluster, '_memory_usage', 0)
        new_experiences = storage_results["success_count"]
        
        cluster._memory_usage = old_memory_usage + new_experiences
        
        return {
            "updated": True,
            "old_memory_usage": old_memory_usage,
            "new_memory_usage": cluster._memory_usage,
            "experiences_added": new_experiences
        }
    
    # Вспомогательные методы для _memory_retrieve_similar
    def _search_across_nodes(self, cluster: ClusterContainer,
                           query_vector: np.ndarray,
                           threshold: float,
                           max_results: int) -> Dict[str, Any]:
        """Поиск похожих воспоминаний по всем узлам"""
        all_matches = []
        nodes_searched = 0
        
        for item in cluster.get_active_items():
            if isinstance(item, NodeEntity) and item.isactive:
                nodes_searched += 1
                
                try:
                    # Выполняем поиск в памяти узла
                    node_matches = item.query_memory(
                        query_vector=query_vector,
                        k=min(5, max_results),
                        threshold=threshold
                    )
                    
                    for pattern, similarity in node_matches:
                        all_matches.append({
                            "node": item.name,
                            "pattern": pattern,
                            "similarity": similarity,
                            "timestamp": pattern.timestamp,
                            "confidence": pattern.confidence
                        })
                        
                except Exception as e:
                    logger.error(f"Error searching memory in node '{item.name}': {e}")
                    continue
        
        return {
            "nodes_searched": nodes_searched,
            "total_matches": len(all_matches),
            "all_matches": all_matches
        }
    
    def _apply_context_filter(self, search_results: Dict[str, Any],
                            context: Dict[str, Any],
                            threshold: float) -> List[Dict[str, Any]]:
        """Применение контекстной фильтрации к результатам поиска"""
        if not search_results["all_matches"]:
            return []
        
        filtered = []
        
        for match in search_results["all_matches"]:
            # Базовый порог схожести
            if match["similarity"] < threshold:
                continue
            
            # Контекстная фильтрация
            context_score = self._calculate_context_score(match, context)
            
            if context_score > 0.3:  # Минимальный контекстный score
                match["context_score"] = context_score
                match["combined_score"] = (match["similarity"] * 0.7 + context_score * 0.3)
                filtered.append(match)
        
        return filtered
    
    def _calculate_context_score(self, match: Dict[str, Any],
                               context: Dict[str, Any]) -> float:
        """Расчет контекстного score"""
        if not context:
            return 0.5  # Нейтральный score при отсутствии контекста
        
        context_factors = []
        
        # Фактор 1: Временная релевантность
        if "time_context" in context and match.get("timestamp"):
            time_diff = abs((datetime.now() - match["timestamp"]).total_seconds())
            time_relevance = 1.0 / (1.0 + time_diff / 3600)  # Затухание по часам
            context_factors.append(time_relevance * 0.4)
        
        # Фактор 2: Тематическая релевантность
        if "topic" in context and match["pattern"].metadata:
            metadata = match["pattern"].metadata
            topic_match = context["topic"].lower() in str(metadata).lower()
            context_factors.append(0.3 if topic_match else 0.1)
        
        # Фактор 3: Уверенность в памяти
        confidence = match.get("confidence", 0.5)
        context_factors.append(confidence * 0.3)
        
        return np.mean(context_factors) if context_factors else 0.5
    
    def _apply_temporal_weighting(self, filtered_matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Применение временного взвешивания к результатам"""
        if not filtered_matches:
            return []
        
        current_time = datetime.now()
        
        for match in filtered_matches:
            if "timestamp" in match and match["timestamp"]:
                # Вычисляем возраст воспоминания
                age_seconds = (current_time - match["timestamp"]).total_seconds()
                
                # Вес на основе возраста (новые воспоминания имеют больший вес)
                # Используем экспоненциальное затухание с half-life в 24 часа
                age_weight = np.exp(-age_seconds / (24 * 3600 * np.log(2)))
                
                # Обновляем combined score с учетом возраста
                match["age_weight"] = age_weight
                match["temporally_adjusted_score"] = match.get("combined_score", 0.5) * age_weight
        
        return filtered_matches
    
    def _rank_and_select_results(self, weighted_matches: List[Dict[str, Any]],
                               max_results: int) -> List[Dict[str, Any]]:
        """Ранжирование и выборка финальных результатов"""
        if not weighted_matches:
            return []
        
        # Сортируем по temporally_adjusted_score
        sorted_matches = sorted(
            weighted_matches,
            key=lambda x: x.get("temporally_adjusted_score", 0),
            reverse=True
        )
        
        # Выбираем топ-N результатов
        selected = sorted_matches[:max_results]
        
        # Форматируем результаты
        formatted_results = []
        for match in selected:
            formatted = {
                "node": match["node"],
                "similarity": match["similarity"],
                "confidence": match.get("confidence", 0.5),
                "timestamp": match["timestamp"].isoformat() if match.get("timestamp") else "unknown",
                "age_weight": match.get("age_weight", 1.0),
                "adjusted_score": match.get("temporally_adjusted_score", 0),
                "pattern_info": {
                    "label": match["pattern"].label,
                    "metadata": match["pattern"].metadata
                }
            }
            formatted_results.append(formatted)
        
        return formatted_results
    
    def _analyze_retrieval_patterns(self, query_vector: np.ndarray,
                                  results: List[Dict[str, Any]],
                                  context: Dict[str, Any]) -> Dict[str, Any]:
        """Анализ паттернов поиска"""
        analysis = {
            "query_complexity": np.linalg.norm(query_vector) if hasattr(query_vector, 'shape') else 0,
            "result_distribution": {},
            "context_influence": 0.0,
            "retrieval_quality": 0.0
        }
        
        if results:
            # Анализ распределения по узлам
            node_distribution = {}
            for result in results:
                node = result["node"]
                node_distribution[node] = node_distribution.get(node, 0) + 1
            
            analysis["result_distribution"] = {
                "unique_nodes": len(node_distribution),
                "node_distribution": node_distribution,
                "concentration": max(node_distribution.values()) / len(results) if results else 0
            }
            
            # Анализ качества поиска
            similarities = [r["similarity"] for r in results]
            analysis["retrieval_quality"] = np.mean(similarities) if similarities else 0
            
            # Влияние контекста
            if context:
                context_scores = [r.get("context_score", 0.5) for r in results if "context_score" in r]
                analysis["context_influence"] = np.mean(context_scores) if context_scores else 0.5
        
        return analysis
    
    # Вспомогательные методы для _memory_consolidate_sleep
    def _offline_replay(self, cluster: ClusterContainer,
                      mode: str) -> Dict[str, Any]:
        """Оффлайн replay важных воспоминаний"""
        replay_results = {
            "experiences_replayed": 0,
            "successful_replays": 0,
            "failed_replays": 0,
            "replay_benefit": 0.0,
            "replayed_experiences": []
        }
        
        # Выбираем воспоминания для replay
        experiences_to_replay = self._select_experiences_for_replay(cluster, mode)
        
        for exp_info in experiences_to_replay[:10]:  # Ограничиваем 10 воспоминаниями
            replay_results["experiences_replayed"] += 1
            
            try:
                # Извлекаем воспоминание
                memory_pattern = self._retrieve_memory_pattern(cluster, exp_info)
                
                if memory_pattern:
                    # Выполняем replay (повторное обучение)
                    replay_success = self._execute_memory_replay(
                        cluster, memory_pattern, exp_info
                    )
                    
                    if replay_success:
                        replay_results["successful_replays"] += 1
                        replay_results["replay_benefit"] += exp_info.get("importance", 0.1)
                        replay_results["replayed_experiences"].append({
                            "experience_id": exp_info.get("experience_id", "unknown"),
                            "importance": exp_info.get("importance", 0),
                            "replay_success": True
                        })
                    else:
                        replay_results["failed_replays"] += 1
                else:
                    replay_results["failed_replays"] += 1
                    
            except Exception as e:
                logger.error(f"Error during memory replay: {e}")
                replay_results["failed_replays"] += 1
        
        # Вычисляем общий benefit
        if replay_results["experiences_replayed"] > 0:
            replay_results["replay_benefit"] /= replay_results["experiences_replayed"]
        
        return replay_results
    
    def _select_experiences_for_replay(self, cluster: ClusterContainer,
                                     mode: str) -> List[Dict[str, Any]]:
        """Выбор воспоминаний для replay"""
        experiences = []
        
        # Собираем информацию о воспоминаниях из буфера опыта
        for exp_record in list(self._experience_buffer)[-50:]:  # Последние 50 опытов
            if exp_record.get("storage_success", False):
                importance = exp_record.get("importance", 0.5)
                
                # Фильтрация в зависимости от mode
                if mode == "standard" and importance > 0.3:
                    experiences.append({
                        "experience_id": exp_record["experience_id"],
                        "importance": importance,
                        "timestamp": exp_record["timestamp"],
                        "nodes": exp_record.get("nodes_used", [])
                    })
                elif mode == "aggressive" and importance > 0.1:
                    experiences.append({
                        "experience_id": exp_record["experience_id"],
                        "importance": importance,
                        "timestamp": exp_record["timestamp"],
                        "nodes": exp_record.get("nodes_used", [])
                    })
                elif mode == "conservative" and importance > 0.7:
                    experiences.append(exp_record)
        
        # Сортируем по важности
        experiences.sort(key=lambda x: x.get("importance", 0), reverse=True)
        
        return experiences
    
    def _retrieve_memory_pattern(self, cluster: ClusterContainer,
                               exp_info: Dict[str, Any]) -> Optional[Any]:
        """Извлечение паттерна памяти по информации об опыте"""
        # Ищем паттерн в узлах, где он был сохранен
        for node_name in exp_info.get("nodes", []):
            if cluster.has_item(node_name):
                node = cluster.get(node_name)
                if isinstance(node, NodeEntity):
                    # Ищем паттерн по label
                    label = f"exp_{exp_info['experience_id']}"
                    
                    # Упрощенный поиск - в реальности нужен более сложный механизм
                    for pattern in node._memory.patterns:
                        if pattern.label == label:
                            return pattern
        
        return None
    
    def _execute_memory_replay(self, cluster: ClusterContainer,
                             memory_pattern: Any,
                             exp_info: Dict[str, Any]) -> bool:
        """Выполнение replay воспоминания"""
        try:
            # Создаем synthetic input на основе паттерна
            input_vector = memory_pattern.vector
            
            # Находим подходящий узел для replay
            replay_node = self._select_replay_node(cluster, exp_info)
            
            if replay_node:
                # Выполняем обучение на основе паттерна
                result = replay_node.learn_from_experience(
                    input_vector=input_vector,
                    target_output=memory_pattern.vector * 0.9,  # Легкая модификация
                    reward=exp_info.get("importance", 0.5) * 0.8
                )
                
                return result.get("status") != "inactive"
        
        except Exception as e:
            logger.error(f"Error executing memory replay: {e}")
        
        return False
    
    def _select_replay_node(self, cluster: ClusterContainer,
                          exp_info: Dict[str, Any]) -> Optional[NodeEntity]:
        """Выбор узла для replay"""
        # Предпочитаем оригинальные узлы хранения
        for node_name in exp_info.get("nodes", []):
            if cluster.has_item(node_name):
                node = cluster.get(node_name)
                if isinstance(node, NodeEntity) and node.isactive and node.energy_level > 0.3:
                    return node
        
        # Или любой активный узел с достаточной энергией
        for item in cluster.get_active_items():
            if isinstance(item, NodeEntity) and item.energy_level > 0.5:
                return item
        
        return None
    
    def _reorganize_memory_structure(self, cluster: ClusterContainer,
                                   replay_results: Dict[str, Any]) -> Dict[str, Any]:
        """Реорганизация структуры памяти"""
        reorganization = {
            "nodes_optimized": 0,
            "patterns_reorganized": 0,
            "structure_improvement": 0.0
        }
        
        for item in cluster.get_active_items():
            if isinstance(item, NodeEntity):
                # Проверяем, нужна ли оптимизация памяти узла
                memory_stats = item._memory.get_stats()
                pattern_count = memory_stats.get("pattern_count", 0)
                
                if pattern_count > item.memory_capacity * 0.8:
                    # Узел почти заполнен - выполняем очистку
                    old_count = pattern_count
                    self._optimize_node_memory(item)
                    new_count = item._memory.get_stats().get("pattern_count", 0)
                    
                    if new_count < old_count:
                        reorganization["nodes_optimized"] += 1
                        reorganization["patterns_reorganized"] += (old_count - new_count)
                        reorganization["structure_improvement"] += (old_count - new_count) / max(1, old_count)
        
        if reorganization["nodes_optimized"] > 0:
            reorganization["structure_improvement"] /= reorganization["nodes_optimized"]
        
        return reorganization
    
    def _optimize_node_memory(self, node: NodeEntity):
        """Оптимизация памяти узла"""
        # Удаляем наименее используемые паттерны
        patterns = node._memory.patterns
        if len(patterns) > node.memory_capacity * 0.7:
            # Сортируем по access_count (наименее используемые сначала)
            patterns.sort(key=lambda p: p.access_count)
            
            # Удаляем нижние 20%
            remove_count = int(len(patterns) * 0.2)
            for _ in range(remove_count):
                if patterns:
                    patterns.pop(0)
            
            # Перестраиваем индекс
            node._memory._rebuild_index()
    
    def _compress_knowledge(self, cluster: ClusterContainer,
                          strength: float) -> Dict[str, Any]:
        """Сжатие знаний через упрощенный autoencoder"""
        compression_results = {
            "nodes_processed": 0,
            "patterns_compressed": 0,
            "compression_ratio": 1.0,
            "quality_preserved": 0.0
        }
        
        total_original_size = 0
        total_compressed_size = 0
        quality_scores = []
        
        for item in cluster.get_active_items():
            if isinstance(item, NodeEntity) and item._memory.patterns:
                compression_results["nodes_processed"] += 1
                
                node_original_size = 0
                node_compressed_size = 0
                
                for pattern in item._memory.patterns:
                    # Упрощенное сжатие: уменьшаем точность
                    original_vector = pattern.vector
                    node_original_size += original_vector.nbytes
                    
                    if strength > 0.3:
                        # Применяем сжатие
                        compressed_vector = self._apply_simple_compression(
                            original_vector, strength
                        )
                        
                        # Обновляем паттерн
                        pattern.vector = compressed_vector
                        compression_results["patterns_compressed"] += 1
                        
                        node_compressed_size += compressed_vector.nbytes
                        
                        # Оцениваем сохранение качества
                        similarity = np.dot(original_vector, compressed_vector) / (
                            np.linalg.norm(original_vector) * np.linalg.norm(compressed_vector)
                        )
                        quality_scores.append(similarity)
                    else:
                        node_compressed_size += original_vector.nbytes
                
                total_original_size += node_original_size
                total_compressed_size += node_compressed_size
        
        if total_original_size > 0:
            compression_results["compression_ratio"] = total_original_size / max(1, total_compressed_size)
        
        if quality_scores:
            compression_results["quality_preserved"] = np.mean(quality_scores)
        
        return compression_results
    
    def _apply_simple_compression(self, vector: np.ndarray,
                                strength: float) -> np.ndarray:
        """Упрощенное сжатие вектора"""
        if strength < 0.1:
            return vector
        
        # Уменьшаем точность
        if strength > 0.7:
            # Сильное сжатие: float32 -> float16
            return vector.astype(np.float16).astype(np.float32)
        elif strength > 0.4:
            # Умеренное сжатие: уменьшаем значения
            scale = 0.8
            return vector * scale
        else:
            # Легкое сжатие: добавляем небольшой шум для regularization
            noise = np.random.randn(*vector.shape).astype(np.float32) * 0.01
            return vector + noise
    
    def _generate_dream_patterns(self, cluster: ClusterContainer,
                               creativity_level: float,
                               replay_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Генерация dream-like паттернов"""
        dream_patterns = []
        
        if creativity_level < 0.1:
            return dream_patterns
        
        # Выбираем узлы для генерации снов
        creative_nodes = self._select_creative_nodes(cluster, creativity_level)
        
        for node_name in creative_nodes[:3]:  # Ограничиваем 3 узлами
            if cluster.has_item(node_name):
                node = cluster.get(node_name)
                if isinstance(node, NodeEntity) and node._memory.patterns:
                    
                    # Генерируем dream паттерны
                    for _ in range(int(creativity_level * 3)):  # 1-3 паттерна на узел
                        dream_pattern = self._generate_single_dream(node)
                        if dream_pattern:
                            dream_patterns.append({
                                "node": node_name,
                                "dream_pattern": dream_pattern,
                                "creativity_level": creativity_level
                            })
        
        return dream_patterns
    
    def _select_creative_nodes(self, cluster: ClusterContainer,
                             creativity_level: float) -> List[str]:
        """Выбор креативных узлов для генерации снов"""
        creative_nodes = []
        
        for item in cluster.get_active_items():
            if isinstance(item, NodeEntity):
                # Узлы с абстрактной специализацией более креативны
                if item.specialization in ["abstract", "integrative", "visual"]:
                    creative_nodes.append(item.name)
                # Узлы с высокой энергией также могут быть креативными
                elif item.energy_level > 0.8 and creativity_level > 0.5:
                    creative_nodes.append(item.name)
        
        return creative_nodes
    
    def _generate_single_dream(self, node: NodeEntity) -> Optional[np.ndarray]:
        """Генерация одного dream паттерна"""
        if not node._memory.patterns:
            return None
        
        try:
            # Выбираем случайные паттерны для комбинации
            num_patterns = min(3, len(node._memory.patterns))
            selected_patterns = random.sample(node._memory.patterns, num_patterns)
            
            # Комбинируем паттерны
            combined_vector = np.zeros_like(selected_patterns[0].vector)
            weights = np.random.rand(num_patterns)
            weights = weights / weights.sum()
            
            for i, pattern in enumerate(selected_patterns):
                combined_vector += pattern.vector * weights[i]
            
            # Добавляем креативный шум
            creativity_noise = np.random.randn(*combined_vector.shape).astype(np.float32) * 0.1
            dream_vector = combined_vector + creativity_noise
            
            # Нормализуем
            norm = np.linalg.norm(dream_vector)
            if norm > 0:
                dream_vector = dream_vector / norm
            
            return dream_vector
            
        except Exception as e:
            logger.error(f"Error generating dream pattern: {e}")
            return None
    
    def _cleanup_memory(self, cluster: ClusterContainer,
                      reorganization_results: Dict[str, Any]) -> Dict[str, Any]:
        """Очистка и оптимизация памяти"""
        cleanup_results = {
            "memory_freed_mb": 0.0,
            "fragmentation_reduced": 0.0,
            "performance_improvement": 0.0
        }
        
        # Очищаем буферы и кэши
        old_experience_buffer_size = len(self._experience_buffer)
        old_retrieval_patterns_size = len(self._retrieval_patterns)
        
        # Удаляем старые записи (старше 7 дней)
        cutoff_time = datetime.now() - timedelta(days=7)
        
        # Очищаем буфер опыта
        self._experience_buffer = deque(
            [exp for exp in self._experience_buffer if exp["timestamp"] > cutoff_time],
            maxlen=1000
        )
        
        # Очищаем паттерны поиска
        self._retrieval_patterns = {
            k: v for k, v in self._retrieval_patterns.items()
            if v["last_seen"] > cutoff_time
        }
        
        # Оцениваем освобожденную память
        experiences_removed = old_experience_buffer_size - len(self._experience_buffer)
        patterns_removed = old_retrieval_patterns_size - len(self._retrieval_patterns)
        
        # Примерная оценка (в реальности было бы точнее)
        cleanup_results["memory_freed_mb"] = (
            experiences_removed * 0.01 +  # 10KB на опыт
            patterns_removed * 0.005      # 5KB на паттерн
        )
        
        # Оцениваем улучшение производительности
        if old_experience_buffer_size > 0:
            cleanup_results["performance_improvement"] = min(
                0.3, experiences_removed / old_experience_buffer_size * 0.5
            )
        
        return cleanup_results
    
    def _update_consolidation_schedule(self, cluster: ClusterContainer,
                                     replay_results: Dict[str, Any],
                                     compression_results: Dict[str, Any]) -> Dict[str, Any]:
        """Обновление расписания консолидации"""
        # Анализируем результаты консолидации
        consolidation_quality = (
            replay_results.get("replay_benefit", 0) * 0.6 +
            compression_results.get("quality_preserved", 0) * 0.4
        )
        
        # Определяем рекомендуемый интервал до следующей консолидации
        if consolidation_quality > 0.7:
            next_interval = "48h"  # Высокое качество - можно ждать дольше
        elif consolidation_quality > 0.4:
            next_interval = "24h"  # Среднее качество - стандартный интервал
        else:
            next_interval = "12h"  # Низкое качество - нужно чаще
        
        # Обновляем расписание
        schedule_entry = {
            "timestamp": datetime.now(),
            "cluster": cluster.name,
            "consolidation_quality": consolidation_quality,
            "next_recommended": next_interval
        }
        
        # Храним только последние 10 записей
        self._consolidation_schedule = list(self._consolidation_schedule)[-9:] + [schedule_entry]
        
        return {
            "updated": True,
            "consolidation_quality": consolidation_quality,
            "next_recommended": next_interval,
            "schedule_entries": len(self._consolidation_schedule)
        }
    
    # Вспомогательные методы для _memory_analyze
    def _analyze_memory_comprehensive(self, cluster: ClusterContainer) -> Dict[str, Any]:
        """Комплексный анализ памяти кластера"""
        analysis = {
            "memory_capacity": {},
            "usage_statistics": {},
            "performance_metrics": {},
            "health_indicators": {},
            "recommendations": []
        }
        
        # Анализ capacity
        total_capacity = 0
        total_used = 0
        
        for item in cluster.get_active_items():
            if isinstance(item, NodeEntity):
                stats = item._memory.get_stats()
                total_capacity += item.memory_capacity
                total_used += stats.get("pattern_count", 0)
        
        analysis["memory_capacity"] = {
            "total_capacity": total_capacity,
            "total_used": total_used,
            "usage_percentage": total_used / max(1, total_capacity) * 100,
            "nodes_with_memory": len([item for item in cluster.get_active_items() 
                                    if isinstance(item, NodeEntity)])
        }
        
        # Статистика использования
        retrieval_stats = self._analyze_retrieval_statistics()
        storage_stats = self._analyze_storage_statistics()
        
        analysis["usage_statistics"] = {
            "retrieval": retrieval_stats,
            "storage": storage_stats,
            "consolidation_sessions": len(self._consolidation_schedule)
        }
        
        # Метрики производительности
        analysis["performance_metrics"] = self._calculate_memory_performance(
            retrieval_stats, storage_stats
        )
        
        # Индикаторы здоровья
        analysis["health_indicators"] = self._assess_memory_health(
            analysis["memory_capacity"],
            analysis["performance_metrics"]
        )
        
        # Рекомендации
        analysis["recommendations"] = self._generate_memory_recommendations(analysis)
        
        return analysis
    
    def _analyze_retrieval_statistics(self) -> Dict[str, Any]:
        """Анализ статистики поиска"""
        if not self._retrieval_patterns:
            return {"total_queries": 0, "average_similarity": 0, "unique_queries": 0}
        
        total_queries = sum(pattern["count"] for pattern in self._retrieval_patterns.values())
        avg_similarity = np.mean([pattern.get("average_similarity", 0) 
                                for pattern in self._retrieval_patterns.values()])
        
        return {
            "total_queries": total_queries,
            "average_similarity": avg_similarity,
            "unique_queries": len(self._retrieval_patterns),
            "query_frequency": total_queries / max(1, len(self._retrieval_patterns))
        }
    
    def _analyze_storage_statistics(self) -> Dict[str, Any]:
        """Анализ статистики хранения"""
        if not self._experience_buffer:
            return {"total_experiences": 0, "average_importance": 0, "storage_success_rate": 0}
        
        total_experiences = len(self._experience_buffer)
        avg_importance = np.mean([exp.get("importance", 0) for exp in self._experience_buffer])
        success_count = sum(1 for exp in self._experience_buffer if exp.get("storage_success", False))
        
        return {
            "total_experiences": total_experiences,
            "average_importance": avg_importance,
            "storage_success_rate": success_count / max(1, total_experiences),
            "recent_activity": len([exp for exp in self._experience_buffer 
                                  if exp["timestamp"] > datetime.now() - timedelta(hours=1)])
        }
    
    def _calculate_memory_performance(self, retrieval_stats: Dict[str, Any],
                                    storage_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Расчет метрик производительности памяти"""
        performance = {
            "retrieval_efficiency": retrieval_stats.get("average_similarity", 0),
            "storage_efficiency": storage_stats.get("storage_success_rate", 0),
            "memory_throughput": 0,
            "latency_estimate": 0
        }
        
        # Оценка throughput (операций в час)
        recent_hours = 24
        recent_experiences = len([exp for exp in self._experience_buffer 
                                if exp["timestamp"] > datetime.now() - timedelta(hours=recent_hours)])
        
        performance["memory_throughput"] = recent_experiences / recent_hours
        
        # Оценка latency на основе сложности операций
        if retrieval_stats.get("total_queries", 0) > 0:
            # Предполагаемая latency в секундах
            performance["latency_estimate"] = 0.1 + (1.0 - performance["retrieval_efficiency"]) * 0.5
        
        return performance
    
    def _assess_memory_health(self, capacity_info: Dict[str, Any],
                            performance_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Оценка здоровья памяти"""
        health_indicators = {
            "capacity_health": 0.0,
            "performance_health": 0.0,
            "overall_health": 0.0,
            "issues_detected": []
        }
        
        # Здоровье capacity
        usage_percentage = capacity_info.get("usage_percentage", 0)
        if usage_percentage > 90:
            health_indicators["capacity_health"] = 0.3
            health_indicators["issues_detected"].append("High memory usage (>90%)")
        elif usage_percentage > 70:
            health_indicators["capacity_health"] = 0.6
            health_indicators["issues_detected"].append("Moderate memory usage (>70%)")
        else:
            health_indicators["capacity_health"] = 0.9
        
        # Здоровье производительности
        retrieval_efficiency = performance_metrics.get("retrieval_efficiency", 0)
        storage_efficiency = performance_metrics.get("storage_efficiency", 0)
        
        performance_score = (retrieval_efficiency * 0.6 + storage_efficiency * 0.4)
        
        if performance_score > 0.8:
            health_indicators["performance_health"] = 0.9
        elif performance_score > 0.6:
            health_indicators["performance_health"] = 0.7
        else:
            health_indicators["performance_health"] = 0.4
            health_indicators["issues_detected"].append("Low memory performance")
        
        # Общее здоровье
        health_indicators["overall_health"] = (
            health_indicators["capacity_health"] * 0.4 +
            health_indicators["performance_health"] * 0.6
        )
        
        return health_indicators
    
    def _generate_memory_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Генерация рекомендаций по памяти"""
        recommendations = []
        
        capacity_info = analysis.get("memory_capacity", {})
        health_indicators = analysis.get("health_indicators", {})
        
        # Рекомендации по capacity
        if capacity_info.get("usage_percentage", 0) > 80:
            recommendations.append("Consider increasing memory capacity or implementing more aggressive compression")
        
        if capacity_info.get("usage_percentage", 0) < 20:
            recommendations.append("Memory underutilized - consider storing more experiences or reducing capacity")
        
        # Рекомендации по производительности
        if health_indicators.get("performance_health", 0) < 0.6:
            recommendations.append("Schedule memory consolidation to improve performance")
        
        if analysis.get("usage_statistics", {}).get("retrieval", {}).get("average_similarity", 0) < 0.6:
            recommendations.append("Improve indexing or increase memory precision for better retrieval")
        
        # Общие рекомендации
        if not recommendations:
            recommendations.append("Memory system is operating within optimal parameters")
        
        return recommendations
    
    def _analyze_memory_performance(self, cluster: ClusterContainer) -> Dict[str, Any]:
        """Анализ производительности памяти"""
        performance = {
            "retrieval_performance": self._analyze_retrieval_performance(),
            "storage_performance": self._analyze_storage_performance(),
            "consolidation_performance": self._analyze_consolidation_performance(),
            "bottlenecks": self._identify_memory_bottlenecks(cluster)
        }
        
        return performance
    
    def _analyze_retrieval_performance(self) -> Dict[str, Any]:
        """Анализ производительности поиска"""
        if not self._retrieval_patterns:
            return {"status": "no_data", "average_latency": 0, "success_rate": 0}
        
        # Анализируем последние 50 поисков
        recent_patterns = list(self._retrieval_patterns.values())[-50:]
        
        avg_similarity = np.mean([p.get("average_similarity", 0) for p in recent_patterns])
        query_frequency = len(recent_patterns) / 24.0  # Запросов в час
        
        return {
            "status": "active",
            "average_similarity": avg_similarity,
            "query_frequency_per_hour": query_frequency,
            "unique_queries_last_24h": len(recent_patterns),
            "performance_score": min(1.0, avg_similarity * 0.7 + min(1.0, query_frequency / 10) * 0.3)
        }
    
    def _analyze_storage_performance(self) -> Dict[str, Any]:
        """Анализ производительности хранения"""
        if not self._experience_buffer:
            return {"status": "no_data", "storage_rate": 0, "success_rate": 0}
        
        recent_experiences = [exp for exp in self._experience_buffer 
                            if exp["timestamp"] > datetime.now() - timedelta(hours=24)]
        
        if not recent_experiences:
            return {"status": "no_recent_data", "storage_rate": 0, "success_rate": 0}
        
        storage_rate = len(recent_experiences) / 24.0  # Опытов в час
        success_rate = sum(1 for exp in recent_experiences if exp.get("storage_success", False)) / len(recent_experiences)
        avg_importance = np.mean([exp.get("importance", 0) for exp in recent_experiences])
        
        return {
            "status": "active",
            "storage_rate_per_hour": storage_rate,
            "success_rate": success_rate,
            "average_importance": avg_importance,
            "performance_score": min(1.0, success_rate * 0.5 + avg_importance * 0.5)
        }
    
    def _analyze_consolidation_performance(self) -> Dict[str, Any]:
        """Анализ производительности консолидации"""
        if not self._consolidation_schedule:
            return {"status": "no_data", "average_quality": 0, "frequency": 0}
        
        recent_sessions = [sess for sess in self._consolidation_schedule 
                         if sess["timestamp"] > datetime.now() - timedelta(days=7)]
        
        if not recent_sessions:
            return {"status": "no_recent_data", "average_quality": 0, "frequency": 0}
        
        avg_quality = np.mean([sess.get("consolidation_quality", 0) for sess in recent_sessions])
        avg_duration = np.mean([sess.get("duration_seconds", 0) for sess in recent_sessions])
        frequency = len(recent_sessions) / 7.0  # Сессий в день
        
        return {
            "status": "active",
            "average_quality": avg_quality,
            "average_duration_seconds": avg_duration,
            "frequency_per_day": frequency,
            "performance_score": min(1.0, avg_quality * 0.6 + min(1.0, frequency) * 0.4)
        }
    
    def _identify_memory_bottlenecks(self, cluster: ClusterContainer) -> List[str]:
        """Идентификация узких мест в памяти"""
        bottlenecks = []
        
        # Проверяем узлы с высокой загрузкой памяти
        for item in cluster.get_active_items():
            if isinstance(item, NodeEntity):
                memory_stats = item._memory.get_stats()
                pattern_count = memory_stats.get("pattern_count", 0)
                capacity = item.memory_capacity
                
                if pattern_count > capacity * 0.9:
                    bottlenecks.append(f"Node '{item.name}' memory almost full ({pattern_count}/{capacity})")
                
                # Проверяем скорость поиска (упрощенная)
                if pattern_count > 500 and item.activation_count < 10:
                    bottlenecks.append(f"Node '{item.name}' has many patterns but low activity")
        
        # Проверяем общую статистику
        retrieval_stats = self._analyze_retrieval_statistics()
        if retrieval_stats.get("average_similarity", 0) < 0.5:
            bottlenecks.append("Low retrieval similarity - may indicate poor memory organization")
        
        if not bottlenecks:
            bottlenecks.append("No significant bottlenecks identified")
        
        return bottlenecks
    
    def _analyze_memory_content(self, cluster: ClusterContainer) -> Dict[str, Any]:
        """Анализ содержимого памяти"""
        content_analysis = {
            "pattern_distribution": {},
            "content_types": {},
            "temporal_distribution": {},
            "quality_metrics": {}
        }
        
        # Анализ распределения паттернов по узлам
        node_distribution = {}
        for item in cluster.get_active_items():
            if isinstance(item, NodeEntity):
                memory_stats = item._memory.get_stats()
                pattern_count = memory_stats.get("pattern_count", 0)
                node_distribution[item.name] = {
                    "patterns": pattern_count,
                    "specialization": item.specialization
                }
        
        content_analysis["pattern_distribution"] = node_distribution
        
        # Анализ типов содержимого (на основе специализации)
        content_types = defaultdict(int)
        for node_info in node_distribution.values():
            content_types[node_info["specialization"]] += node_info["patterns"]
        
        content_analysis["content_types"] = dict(content_types)
        
        # Временное распределение (на основе буфера опыта)
        if self._experience_buffer:
            time_buckets = defaultdict(int)
            for exp in self._experience_buffer:
                hour = exp["timestamp"].hour
                time_bucket = f"{hour:02d}:00-{hour:02d}:59"
                time_buckets[time_bucket] += 1
            
            content_analysis["temporal_distribution"] = dict(time_buckets)
        
        # Метрики качества
        quality_metrics = {
            "average_pattern_age": self._calculate_average_pattern_age(cluster),
            "memory_diversity": len(content_types) / max(1, len(node_distribution)),
            "storage_efficiency": self._calculate_storage_efficiency(cluster)
        }
        
        content_analysis["quality_metrics"] = quality_metrics
        
        return content_analysis
    
    def _calculate_average_pattern_age(self, cluster: ClusterContainer) -> float:
        """Расчет среднего возраста паттернов"""
        if not self._experience_buffer:
            return 0.0
        
        current_time = datetime.now()
        ages = []
        
        for exp in self._experience_buffer[-100:]:  # Последние 100 опытов
            age_hours = (current_time - exp["timestamp"]).total_seconds() / 3600
            ages.append(age_hours)
        
        return np.mean(ages) if ages else 0.0
    
    def _calculate_storage_efficiency(self, cluster: ClusterContainer) -> float:
        """Расчет эффективности хранения"""
        total_patterns = 0
        total_capacity = 0
        
        for item in cluster.get_active_items():
            if isinstance(item, NodeEntity):
                memory_stats = item._memory.get_stats()
                total_patterns += memory_stats.get("pattern_count", 0)
                total_capacity += item.memory_capacity
        
        if total_capacity == 0:
            return 0.0
        
        efficiency = total_patterns / total_capacity
        
        # Идеальная эффективность - около 70%
        return min(1.0, efficiency / 0.7)