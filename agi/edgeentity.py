"""
AGI Adaptive Edge (Synapse Analog)
Implements a directed, weighted connection between NodeEntity instances
with Hebbian learning, plasticity, and evolutionary capabilities.
"""

from typing import Dict, Any, Optional, List, Union
import numpy as np
from datetime import datetime, timedelta
from enum import Enum
import torch
import torch.nn as nn
from common.base.baseentity import BaseEntity
from common.utils.logging_setup import logger


class ConnectionType(Enum):
    """Types of neural connections"""
    EXCITATORY = "excitatory"    # Усиливающая связь
    INHIBITORY = "inhibitory"    # Ослабляющая связь
    MODULATORY = "modulatory"    # Модулирующая связь
    NEUTRAL = "neutral"          # Нейтральная связь


class LearningRule(Enum):
    """Types of synaptic learning rules"""
    HEBB = "hebbian"             # Hebbian: "cells that fire together, wire together"
    OJA = "oja"                  # Oja's rule (stabilized Hebbian)
    STDP = "stdp"                # Spike-timing dependent plasticity
    BCM = "bcm"                  # Bienenstock-Cooper-Munro rule


class EdgeEntity(BaseEntity):
    """
    AGI Adaptive Edge - directed connection between nodes with adaptive weights.
    
    Implements a biological synapse analog with:
    1. Adaptive weight updated via learning rules
    2. Transmission properties (delay, bandwidth)
    3. Plasticity factors and meta-learning
    4. Evolutionary tracking and scoring
    
    Attributes:
        source_node (str): Name of the source NodeEntity
        target_node (str): Name of the target NodeEntity
        
        # Connection properties
        weight (float): Connection strength (-1.0 to 1.0)
        transmission_delay (float): Signal delay in milliseconds (0-100)
        bandwidth (float): Information capacity (0.0-1.0)
        connection_type (str): Type of connection (excitatory/inhibitory/modulatory/neutral)
        
        # Dynamic characteristics
        usage_frequency (int): How many times this edge was activated
        last_reinforcement_time (datetime): When weight was last updated
        plasticity_factor (float): Speed of weight change (0.0-1.0)
        meta_learning_rate (float): Adaptive learning rate (0.0-0.1)
        
        # Evolutionary markers
        age (datetime): When this edge was created
        evolutionary_score (float): Utility score for the system (0.0-1.0)
        
        # Learning parameters
        learning_rule (str): Type of learning rule (hebbian/oja/stdp/bcm)
        plasticity_params (Dict[str, float]): Parameters for learning rule
        weight_history (List[float]): History of weight changes
    """
    
    # === Declare all public attributes for BaseEntity ===
    source_node: str
    target_node: str
    
    # Connection properties
    weight: float
    transmission_delay: float
    bandwidth: float
    connection_type: str
    
    # Dynamic characteristics
    usage_frequency: int
    last_reinforcement_time: datetime
    plasticity_factor: float
    meta_learning_rate: float
    
    # Evolutionary markers
    age: datetime
    evolutionary_score: float
    
    # Learning parameters
    learning_rule: str
    plasticity_params: Dict[str, float]
    weight_history: List[float]
    
    def __init__(self,
                 name: str,
                 source_node: str,
                 target_node: str,
                 weight: float = 0.5,
                 transmission_delay: float = 10.0,
                 bandwidth: float = 1.0,
                 connection_type: Union[str, ConnectionType] = ConnectionType.EXCITATORY,
                 learning_rule: Union[str, LearningRule] = LearningRule.HEBB,
                 plasticity_factor: float = 0.1,
                 meta_learning_rate: float = 0.01,
                 evolutionary_score: float = 0.5,
                 isactive: bool = True,
                 use_cache: bool = False,
                 **kwargs):
        """
        Initialize an EdgeEntity.
        
        Args:
            name: Unique name for the edge
            source_node: Name of source NodeEntity
            target_node: Name of target NodeEntity
            weight: Initial connection strength (-1.0 to 1.0)
            transmission_delay: Signal delay in ms (0-100)
            bandwidth: Information capacity (0.0-1.0)
            connection_type: Type of connection (excitatory/inhibitory/modulatory/neutral)
            learning_rule: Type of learning rule (hebbian/oja/stdp/bcm)
            plasticity_factor: Speed of weight change (0.0-1.0)
            meta_learning_rate: Adaptive learning rate (0.0-0.1)
            evolutionary_score: Initial utility score (0.0-1.0)
            isactive: Whether edge is active
            use_cache: Enable caching for to_dict()
            **kwargs: Additional BaseEntity parameters
        """
        
        # Convert enum to string if needed
        if isinstance(connection_type, ConnectionType):
            connection_type = connection_type.value
        
        if isinstance(learning_rule, LearningRule):
            learning_rule = learning_rule.value
        
        # Initialize BaseEntity first
        super().__init__(
            name=name,
            isactive=isactive,
            use_cache=use_cache,
            **kwargs
        )
        
        # Set edge properties
        self.source_node = source_node
        self.target_node = target_node
        
        # Validate and set weight
        self.weight = self._clamp_weight(weight)
        
        # Connection properties
        self.transmission_delay = max(0.0, min(100.0, transmission_delay))
        self.bandwidth = max(0.0, min(1.0, bandwidth))
        self.connection_type = connection_type
        
        # Dynamic characteristics
        self.usage_frequency = 0
        self.last_reinforcement_time = datetime.now()
        self.plasticity_factor = max(0.0, min(1.0, plasticity_factor))
        self.meta_learning_rate = max(0.0, min(0.1, meta_learning_rate))
        
        # Evolutionary markers
        self.age = datetime.now()
        self.evolutionary_score = max(0.0, min(1.0, evolutionary_score))
        
        # Learning parameters
        self.learning_rule = learning_rule
        self.plasticity_params = self._get_default_plasticity_params(learning_rule)
        self.weight_history = [self.weight]
        
        # Transmission buffer for delayed signals
        self._transmission_buffer: List[Dict[str, Any]] = []
        self._last_transmission_time = datetime.now()
        
        # Statistical tracking
        self._total_transmission_amount = 0.0
        self._average_activation_strength = 0.0
        
        logger.info(f"Created EdgeEntity '{name}' from '{source_node}' to '{target_node}' "
                   f"with weight={weight:.3f}, type={connection_type}")
    
    def _clamp_weight(self, weight: float) -> float:
        """Clamp weight to valid range (-1.0 to 1.0)"""
        return max(-1.0, min(1.0, weight))
    
    def _get_default_plasticity_params(self, learning_rule: str) -> Dict[str, float]:
        """Get default parameters for a learning rule"""
        if learning_rule == LearningRule.HEBB.value:
            return {
                "learning_rate": 0.01,
                "decay_rate": 0.001,
                "max_weight": 1.0,
                "min_weight": -1.0
            }
        elif learning_rule == LearningRule.OJA.value:
            return {
                "learning_rate": 0.01,
                "decay_rate": 0.01,
                "normalization_factor": 0.1
            }
        elif learning_rule == LearningRule.STDP.value:
            return {
                "tau_plus": 20.0,    # Time constant for potentiation
                "tau_minus": 20.0,   # Time constant for depression
                "a_plus": 0.01,      # Potentiation rate
                "a_minus": 0.01,     # Depression rate
                "window_size": 100.0  # STDP window in ms
            }
        elif learning_rule == LearningRule.BCM.value:
            return {
                "learning_rate": 0.01,
                "theta": 0.5,        # Modification threshold
                "tau_theta": 1000.0, # Time constant for threshold adaptation
                "beta": 0.1          # Threshold adaptation rate
            }
        else:
            return {"learning_rate": 0.01}
    
    def transmit(self, 
                 signal_strength: float,
                 timestamp: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Transmit a signal through this edge.
        
        Args:
            signal_strength: Strength of incoming signal (0.0-1.0)
            timestamp: Optional timestamp for transmission
            
        Returns:
            Dictionary with transmission results
        """
        if not self.isactive:
            logger.debug(f"Edge '{self.name}' is inactive, cannot transmit")
            return {
                "status": "inactive",
                "edge": self.name,
                "transmitted": False,
                "output_strength": 0.0,
                "delay": self.transmission_delay
            }
        
        # Apply connection type effect
        if self.connection_type == ConnectionType.INHIBITORY.value:
            effective_strength = -signal_strength * self.weight
        elif self.connection_type == ConnectionType.MODULATORY.value:
            # Modulatory connections modulate existing signals
            effective_strength = signal_strength * (1.0 + self.weight * 0.5)
        else:  # excitatory or neutral
            effective_strength = signal_strength * self.weight
        
        # Apply bandwidth limitation
        if self.bandwidth < 1.0:
            effective_strength *= self.bandwidth
        
        # Apply transmission delay
        transmission_time = timestamp or datetime.now()
        delivery_time = transmission_time + timedelta(milliseconds=self.transmission_delay)
        
        # Store in buffer for delayed delivery
        transmission = {
            "signal": effective_strength,
            "transmission_time": transmission_time,
            "delivery_time": delivery_time,
            "source_strength": signal_strength
        }
        
        self._transmission_buffer.append(transmission)
        
        # Update statistics
        self.usage_frequency += 1
        self._last_transmission_time = transmission_time
        self._total_transmission_amount += abs(effective_strength)
        self._average_activation_strength = (
            (self._average_activation_strength * (self.usage_frequency - 1) + 
             abs(effective_strength)) / self.usage_frequency
        )
        
        logger.debug(f"Edge '{self.name}' transmitting signal {signal_strength:.3f} -> "
                    f"{effective_strength:.3f} with delay {self.transmission_delay}ms")
        
        return {
            "status": "success",
            "edge": self.name,
            "transmitted": True,
            "output_strength": effective_strength,
            "delivery_time": delivery_time,
            "delay": self.transmission_delay,
            "bandwidth_used": self.bandwidth
        }
    
    def get_pending_signals(self, 
                           current_time: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """
        Get signals that are ready for delivery.
        
        Args:
            current_time: Current simulation time (defaults to now)
            
        Returns:
            List of signals ready for delivery
        """
        if not self._transmission_buffer:
            return []
        
        current_time = current_time or datetime.now()
        ready_signals = []
        remaining_signals = []
        
        for transmission in self._transmission_buffer:
            if transmission["delivery_time"] <= current_time:
                ready_signals.append(transmission)
            else:
                remaining_signals.append(transmission)
        
        self._transmission_buffer = remaining_signals
        return ready_signals
    
    def reinforce(self, 
                  reinforcement_strength: float,
                  rule_params: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Reinforce or weaken the connection based on learning rule.
        
        Args:
            reinforcement_strength: Strength of reinforcement (-1.0 to 1.0)
            rule_params: Optional custom parameters for learning rule
            
        Returns:
            Dictionary with reinforcement results
        """
        if not self.isactive:
            return {"status": "inactive", "edge": self.name}
        
        # Use custom params or defaults
        params = rule_params or self.plasticity_params
        
        old_weight = self.weight
        weight_change = 0.0
        
        # Apply learning rule
        if self.learning_rule == LearningRule.HEBB.value:
            weight_change = self._apply_hebbian(reinforcement_strength, params)
        elif self.learning_rule == LearningRule.OJA.value:
            weight_change = self._apply_oja(reinforcement_strength, params)
        elif self.learning_rule == LearningRule.STDP.value:
            weight_change = self._apply_stdp(reinforcement_strength, params)
        elif self.learning_rule == LearningRule.BCM.value:
            weight_change = self._apply_bcm(reinforcement_strength, params)
        else:
            # Default: simple reinforcement
            weight_change = reinforcement_strength * params.get("learning_rate", 0.01)
        
        # Apply plasticity factor and meta-learning rate
        effective_change = (weight_change * self.plasticity_factor * 
                          (1.0 + self.meta_learning_rate))
        
        # Update weight
        new_weight = self._clamp_weight(self.weight + effective_change)
        self.weight = new_weight
        
        # Update tracking
        self.last_reinforcement_time = datetime.now()
        self.weight_history.append(new_weight)
        
        # Update evolutionary score based on usage and stability
        self._update_evolutionary_score()
        
        logger.debug(f"Edge '{self.name}' reinforced: {old_weight:.3f} -> {new_weight:.3f} "
                    f"(change: {effective_change:.4f})")
        
        return {
            "status": "success",
            "edge": self.name,
            "old_weight": old_weight,
            "new_weight": new_weight,
            "weight_change": effective_change,
            "plasticity_factor": self.plasticity_factor,
            "evolutionary_score": self.evolutionary_score
        }
    
    def _apply_hebbian(self, reinforcement: float, params: Dict[str, float]) -> float:
        """Apply Hebbian learning rule"""
        learning_rate = params.get("learning_rate", 0.01)
        decay_rate = params.get("decay_rate", 0.001)
        
        # Hebbian: Δw = η * x * y - λ * w
        # For simplicity, we use reinforcement as correlation signal
        weight_change = learning_rate * reinforcement - decay_rate * self.weight
        return weight_change
    
    def _apply_oja(self, reinforcement: float, params: Dict[str, float]) -> float:
        """Apply Oja's rule (normalized Hebbian)"""
        learning_rate = params.get("learning_rate", 0.01)
        normalization_factor = params.get("normalization_factor", 0.1)
        
        # Oja's rule: Δw = η * (x * y - α * w * y²)
        # Simplified version
        weight_change = learning_rate * (reinforcement - normalization_factor * 
                                       self.weight * reinforcement**2)
        return weight_change
    
    def _apply_stdp(self, reinforcement: float, params: Dict[str, float]) -> float:
        """Apply Spike-timing dependent plasticity"""
        tau_plus = params.get("tau_plus", 20.0)
        tau_minus = params.get("tau_minus", 20.0)
        a_plus = params.get("a_plus", 0.01)
        a_minus = params.get("a_minus", 0.01)
        
        # STDP depends on precise timing, simplified here
        if reinforcement > 0:
            # Potentiation
            weight_change = a_plus * np.exp(-abs(reinforcement) / tau_plus)
        else:
            # Depression
            weight_change = -a_minus * np.exp(-abs(reinforcement) / tau_minus)
        
        return weight_change
    
    def _apply_bcm(self, reinforcement: float, params: Dict[str, float]) -> float:
        """Apply Bienenstock-Cooper-Munro rule"""
        learning_rate = params.get("learning_rate", 0.01)
        theta = params.get("theta", 0.5)
        tau_theta = params.get("tau_theta", 1000.0)
        beta = params.get("beta", 0.1)
        
        # BCM: Δw = η * y * (y - θ) * x
        # θ adapts based on average activity
        activity = abs(reinforcement)
        
        # Update threshold
        theta_change = (activity**2 - theta) / tau_theta
        new_theta = theta + beta * theta_change
        
        # Update weight
        weight_change = learning_rate * reinforcement * (activity - new_theta)
        
        # Store new theta
        self.plasticity_params["theta"] = new_theta
        
        return weight_change
    
    def _update_evolutionary_score(self):
        """Update evolutionary score based on edge performance"""
        
        # Factors for evolutionary score
        usage_factor = min(1.0, self.usage_frequency / 1000.0)  # Normalized usage
        stability_factor = 1.0 - self._calculate_weight_variance()
        age_factor = 1.0 - min(1.0, (datetime.now() - self.age).days / 365.0)
        
        # Weight factors
        if abs(self.weight) > 0.7:
            weight_factor = 1.0  # Strong connections are valuable
        elif abs(self.weight) < 0.1:
            weight_factor = 0.3  # Weak connections are less valuable
        else:
            weight_factor = 0.7
        
        # Calculate composite score
        self.evolutionary_score = (
            0.3 * usage_factor +
            0.3 * stability_factor +
            0.2 * age_factor +
            0.2 * weight_factor
        )
    
    def _calculate_weight_variance(self) -> float:
        """Calculate variance of recent weight changes"""
        if len(self.weight_history) < 2:
            return 0.0
        
        recent_weights = self.weight_history[-min(10, len(self.weight_history)):]
        weights_array = np.array(recent_weights)
        variance = np.var(weights_array)
        
        # Normalize variance (assuming max variance is 1.0 for -1 to 1 range)
        return min(1.0, variance / 0.25)  # 0.25 = variance of uniform [-1, 1]
    
    def strengthen(self, amount: float = 0.1) -> float:
        """
        Strengthen the connection (increase weight).
        
        Args:
            amount: Amount to strengthen (0.0-1.0)
            
        Returns:
            New weight
        """
        new_weight = self._clamp_weight(self.weight + abs(amount))
        self.weight = new_weight
        self.weight_history.append(new_weight)
        self.last_reinforcement_time = datetime.now()
        self.usage_frequency += 1
        
        logger.debug(f"Edge '{self.name}' strengthened: {self.weight:.3f}")
        return new_weight
    
    def weaken(self, amount: float = 0.1) -> float:
        """
        Weaken the connection (decrease weight).
        
        Args:
            amount: Amount to weaken (0.0-1.0)
            
        Returns:
            New weight
        """
        new_weight = self._clamp_weight(self.weight - abs(amount))
        self.weight = new_weight
        self.weight_history.append(new_weight)
        self.last_reinforcement_time = datetime.now()
        self.usage_frequency += 1
        
        logger.debug(f"Edge '{self.name}' weakened: {self.weight:.3f}")
        return new_weight
    
    def update_plasticity(self, 
                         new_plasticity: Optional[float] = None,
                         new_meta_rate: Optional[float] = None) -> Dict[str, float]:
        """
        Update plasticity parameters.
        
        Args:
            new_plasticity: New plasticity factor (0.0-1.0)
            new_meta_rate: New meta-learning rate (0.0-0.1)
            
        Returns:
            Dictionary with updated values
        """
        if new_plasticity is not None:
            self.plasticity_factor = max(0.0, min(1.0, new_plasticity))
        
        if new_meta_rate is not None:
            self.meta_learning_rate = max(0.0, min(0.1, new_meta_rate))
        
        return {
            "plasticity_factor": self.plasticity_factor,
            "meta_learning_rate": self.meta_learning_rate
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive edge statistics"""
        
        weight_variance = self._calculate_weight_variance()
        
        # Calculate age in days
        age_days = (datetime.now() - self.age).total_seconds() / (24 * 3600)
        
        # Calculate reinforcement recency in hours
        recency_hours = (datetime.now() - self.last_reinforcement_time).total_seconds() / 3600
        
        return {
            "name": self.name,
            "source_node": self.source_node,
            "target_node": self.target_node,
            "isactive": self.isactive,
            "weight": self.weight,
            "weight_history_length": len(self.weight_history),
            "weight_variance": weight_variance,
            "transmission_delay": self.transmission_delay,
            "bandwidth": self.bandwidth,
            "connection_type": self.connection_type,
            "usage_frequency": self.usage_frequency,
            "age_days": age_days,
            "evolutionary_score": self.evolutionary_score,
            "plasticity_factor": self.plasticity_factor,
            "meta_learning_rate": self.meta_learning_rate,
            "learning_rule": self.learning_rule,
            "last_reinforcement_hours_ago": recency_hours,
            "pending_signals": len(self._transmission_buffer),
            "total_transmission": self._total_transmission_amount,
            "average_activation": self._average_activation_strength
        }
    
    def is_excitatory(self) -> bool:
        """Check if edge is excitatory"""
        return self.connection_type == ConnectionType.EXCITATORY.value
    
    def is_inhibitory(self) -> bool:
        """Check if edge is inhibitory"""
        return self.connection_type == ConnectionType.INHIBITORY.value
    
    def is_modulatory(self) -> bool:
        """Check if edge is modulatory"""
        return self.connection_type == ConnectionType.MODULATORY.value
    
    def get_connection_strength(self) -> float:
        """
        Get effective connection strength considering all factors.
        
        Returns:
            Effective strength (0.0-1.0)
        """
        base_strength = abs(self.weight)
        
        # Adjust for bandwidth and plasticity
        effective_strength = base_strength * self.bandwidth * (0.5 + 0.5 * self.plasticity_factor)
        
        # Adjust for evolutionary score (successful edges are stronger)
        effective_strength *= (0.5 + 0.5 * self.evolutionary_score)
        
        return min(1.0, effective_strength)
    
    def clear_buffer(self) -> int:
        """
        Clear all pending signals from transmission buffer.
        
        Returns:
            Number of cleared signals
        """
        cleared_count = len(self._transmission_buffer)
        self._transmission_buffer.clear()
        logger.debug(f"Edge '{self.name}' buffer cleared, removed {cleared_count} signals")
        return cleared_count
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert edge to dictionary for serialization"""
        data = super().to_dict()
        
        # Convert datetime fields to ISO strings
        if "last_reinforcement_time" in data and isinstance(data["last_reinforcement_time"], datetime):
            data["last_reinforcement_time"] = data["last_reinforcement_time"].isoformat()
        
        if "age" in data and isinstance(data["age"], datetime):
            data["age"] = data["age"].isoformat()
        
        # Add edge-specific data
        data.update({
            "plasticity_params": self.plasticity_params,
            "weight_history": self.weight_history,
            "transmission_buffer": [
                {
                    "signal": trans["signal"],
                    "transmission_time": trans["transmission_time"].isoformat(),
                    "delivery_time": trans["delivery_time"].isoformat(),
                    "source_strength": trans["source_strength"]
                }
                for trans in self._transmission_buffer
            ],
            "statistics": {
                "total_transmission_amount": self._total_transmission_amount,
                "average_activation_strength": self._average_activation_strength
            }
        })
        
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EdgeEntity':
        """Create edge from dictionary (deserialization)"""
        
        # Extract all fields
        name = data.get("name", "unnamed_edge")
        source_node = data.get("source_node", "unknown")
        target_node = data.get("target_node", "unknown")
        weight = data.get("weight", 0.5)
        transmission_delay = data.get("transmission_delay", 10.0)
        bandwidth = data.get("bandwidth", 1.0)
        connection_type = data.get("connection_type", ConnectionType.EXCITATORY.value)
        learning_rule = data.get("learning_rule", LearningRule.HEBB.value)
        plasticity_factor = data.get("plasticity_factor", 0.1)
        meta_learning_rate = data.get("meta_learning_rate", 0.01)
        evolutionary_score = data.get("evolutionary_score", 0.5)
        isactive = data.get("isactive", True)
        use_cache = data.get("_use_cache", False)
        
        # Handle datetime fields
        age_str = data.get("age")
        if age_str:
            age = datetime.fromisoformat(age_str)
        else:
            age = datetime.now()
        
        last_reinforcement_str = data.get("last_reinforcement_time")
        if last_reinforcement_str:
            last_reinforcement_time = datetime.fromisoformat(last_reinforcement_str)
        else:
            last_reinforcement_time = datetime.now()
        
        usage_frequency = data.get("usage_frequency", 0)
        plasticity_params = data.get("plasticity_params", {})
        weight_history = data.get("weight_history", [])
        
        # Create edge instance
        edge = cls(
            name=name,
            source_node=source_node,
            target_node=target_node,
            weight=weight,
            transmission_delay=transmission_delay,
            bandwidth=bandwidth,
            connection_type=connection_type,
            learning_rule=learning_rule,
            plasticity_factor=plasticity_factor,
            meta_learning_rate=meta_learning_rate,
            evolutionary_score=evolutionary_score,
            isactive=isactive,
            use_cache=use_cache
        )
        
        # Set additional fields
        edge.age = age
        edge.last_reinforcement_time = last_reinforcement_time
        edge.usage_frequency = usage_frequency
        edge.plasticity_params = plasticity_params
        edge.weight_history = weight_history
        
        # Restore transmission buffer
        transmission_buffer_data = data.get("transmission_buffer", [])
        edge._transmission_buffer = []
        for trans_data in transmission_buffer_data:
            transmission = {
                "signal": trans_data["signal"],
                "transmission_time": datetime.fromisoformat(trans_data["transmission_time"]),
                "delivery_time": datetime.fromisoformat(trans_data["delivery_time"]),
                "source_strength": trans_data["source_strength"]
            }
            edge._transmission_buffer.append(transmission)
        
        # Restore statistics
        stats = data.get("statistics", {})
        edge._total_transmission_amount = stats.get("total_transmission_amount", 0.0)
        edge._average_activation_strength = stats.get("average_activation_strength", 0.0)
        
        logger.info(f"Restored EdgeEntity '{edge.name}' from dict")
        return edge
    
    def __repr__(self) -> str:
        """String representation of EdgeEntity"""
        strength = self.get_connection_strength()
        return (
            f"EdgeEntity(name='{self.name}', "
            f"source='{self.source_node}'→'{self.target_node}', "
            f"weight={self.weight:.3f}, "
            f"type={self.connection_type}, "
            f"strength={strength:.3f}, "
            f"usage={self.usage_frequency})"
        )