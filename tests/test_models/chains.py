"""Graph-based chain system for LLM orchestration."""

import asyncio
from enum import Enum
from typing import Any, Callable, Dict, Generic, Optional, Protocol, Set, Tuple, TypeVar, cast, List
from uuid import uuid4

from llmaestro.agents.agent_pool import AgentPool
from llmaestro.core.graph import BaseEdge, BaseGraph, BaseNode
from llmaestro.core.models import LLMResponse
from llmaestro.core.persistence import PersistentModel
from llmaestro.prompts.base import BasePrompt
from llmaestro.prompts.types import PromptMetadata
from llmaestro.llm.responses import ResponseFormat
from pydantic import ConfigDict, Field
from llmaestro.llm.responses import ValidationResult

T = TypeVar("T")
ChainResult = TypeVar("ChainResult")


class NodeType(str, Enum):
    """Types of nodes in the chain graph."""

    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"
    AGENT = "agent"
    VALIDATION = "validation"  # New type for validation nodes


class AgentType(str, Enum):
    """Types of agents that can execute nodes."""

    GENERAL = "general"
    FAST = "fast"
    SPECIALIST = "specialist"


class RetryStrategy(PersistentModel):
    """Configuration for retry behavior."""

    max_retries: int = Field(default=3, ge=0)
    delay: float = Field(default=1.0, ge=0)
    exponential_backoff: bool = Field(default=False)
    max_delay: Optional[float] = Field(default=None, ge=0)

    model_config = ConfigDict(validate_assignment=True)


class ChainMetadata(PersistentModel):
    """Structured metadata for chain components."""

    description: Optional[str] = None
    tags: Set[str] = Field(default_factory=set)
    version: Optional[str] = None
    created_at: Optional[str] = None
    custom_data: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(validate_assignment=True)


class ChainState(PersistentModel):
    """State management for chain execution."""

    status: str = Field(default="pending")
    current_step: Optional[str] = None
    completed_steps: Set[str] = Field(default_factory=set)
    failed_steps: Set[str] = Field(default_factory=set)
    step_results: Dict[str, Any] = Field(default_factory=dict)
    variables: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(validate_assignment=True)


class ChainContext(PersistentModel):
    """Context object passed between chain steps."""

    metadata: ChainMetadata = Field(default_factory=ChainMetadata)
    state: ChainState = Field(default_factory=ChainState)
    variables: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(validate_assignment=True)


class InputTransform(Protocol):
    def __call__(self, context: ChainContext, **kwargs: Any) -> Dict[str, Any]:
        ...


class OutputTransform(Protocol, Generic[T]):
    def __call__(self, response: LLMResponse) -> T:
        ...


class ChainStep(PersistentModel, Generic[T]):
    """Represents a single step in a chain."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    prompt: BasePrompt
    input_transform: Optional[Callable[[ChainContext, Any], Dict[str, Any]]] = None
    output_transform: Optional[Callable[[LLMResponse], T]] = None
    retry_strategy: RetryStrategy = Field(default_factory=RetryStrategy)

    model_config = ConfigDict(validate_assignment=True, arbitrary_types_allowed=True)

    @classmethod
    async def create(
        cls,
        prompt: BasePrompt,
        input_transform: Optional[Callable[[ChainContext, Any], Dict[str, Any]]] = None,
        output_transform: Optional[Callable[[LLMResponse], T]] = None,
        retry_strategy: Optional[RetryStrategy] = None,
    ) -> "ChainStep[T]":
        """Create a new chain step."""
        return cls(
            prompt=prompt,
            input_transform=input_transform,
            output_transform=output_transform,
            retry_strategy=retry_strategy or RetryStrategy(),
        )

    async def execute(
        self,
        agent_pool: AgentPool,
        context: ChainContext,
        **kwargs: Any,
    ) -> T:
        """Execute this chain step using the agent pool."""
        # Transform input using context if needed
        if self.input_transform:
            transformed_data = self.input_transform(context, kwargs)
            # Update prompt with transformed data
            self.prompt = BasePrompt(
                name=self.prompt.name,
                description=self.prompt.description,
                system_prompt=self.prompt.system_prompt.format(**transformed_data),
                user_prompt=self.prompt.user_prompt.format(**transformed_data),
                metadata=self.prompt.metadata,
                variables=self.prompt.variables,
            )

        # Execute prompt
        result = await agent_pool.execute_prompt(self.prompt)

        # Transform output if needed
        if self.output_transform:
            return self.output_transform(result)
        return cast(T, result)


class ChainNode(BaseNode):
    """Represents a single node in the chain graph."""

    step: ChainStep = Field(...)
    node_type: NodeType = Field(...)
    metadata: ChainMetadata = Field(default_factory=ChainMetadata)

    model_config = ConfigDict(validate_assignment=True)


class ChainEdge(BaseEdge):
    """Represents a directed edge between chain nodes."""

    id: str = Field(default_factory=lambda: str(uuid4()), description="Unique identifier for the edge")
    condition: Optional[str] = Field(default=None, description="Optional condition for edge traversal")

    model_config = ConfigDict(validate_assignment=True)


class ChainExecutor:
    """Handles execution of chain nodes with retry logic."""

    @staticmethod
    async def execute_with_retry(
        node: ChainNode, agent_pool: AgentPool, context: ChainContext, retry_strategy: RetryStrategy, **kwargs: Any
    ) -> Any:
        """Execute a node with retry logic."""
        max_retries = retry_strategy.max_retries
        delay = retry_strategy.delay

        for attempt in range(max_retries):
            try:
                return await node.step.execute(agent_pool, context, **kwargs)
            except Exception:
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(delay)
        raise RuntimeError("Node execution failed after all retries")


class ValidationNode(ChainNode):
    """A specialized node for response validation and retry logic."""

    def __init__(
        self,
        response_format: ResponseFormat,
        retry_strategy: Optional[RetryStrategy] = None,
        error_handler: Optional[Callable[[ValidationResult], Dict[str, Any]]] = None,
    ):
        super().__init__(
            node_type=NodeType.VALIDATION,
            step=ChainStep(
                prompt=self._create_retry_prompt(),
                retry_strategy=retry_strategy or RetryStrategy(),
            ),
        )
        self.response_format = response_format
        self.error_handler = error_handler

    async def validate_and_retry(
        self,
        response: LLMResponse,
        agent_pool: AgentPool,
        context: ChainContext,
    ) -> Tuple[bool, Any]:
        """Validate response and handle retries if needed.

        Args:
            response: The LLM response to validate
            agent_pool: Pool of agents for retry execution
            context: Current chain context

        Returns:
            Tuple of (is_valid, final_result)
        """
        validation_result = self.response_format.validate_response(response.content)

        if validation_result.is_valid:
            return True, validation_result.formatted_response

        # Handle retry if needed
        retry_prompt = self.response_format.generate_retry_prompt(validation_result)
        if not retry_prompt:
            if self.error_handler:
                return False, self.error_handler(validation_result)
            return False, validation_result

        # Update prompt with retry context
        self.step.prompt.user_prompt = retry_prompt

        # Execute retry
        retry_response = await self.step.execute(agent_pool, context)
        validation_result.retry_count += 1

        # Validate retry response
        return await self.validate_and_retry(retry_response, agent_pool, context)

    def _create_retry_prompt(self) -> BasePrompt:
        """Create a prompt for retry attempts."""
        return BasePrompt(
            name="validation_retry",
            description="Retry prompt for invalid responses",
            system_prompt=(
                "You are helping to fix an invalid response. "
                "Please address the validation errors and provide a corrected response."
            ),
            user_prompt="{retry_message}",
            metadata=PromptMetadata(type="validation"),
            variables=[],  # No version control for retry prompts
            expected_response=self.response_format,
        )


class ConditionalNode(ChainNode):
    """A node that evaluates conditions and determines the next execution path."""

    def __init__(
        self,
        id: Optional[str] = None,
        conditions: Optional[Dict[str, Callable[[Any], bool]]] = None,
        metadata: Optional[ChainMetadata] = None,
    ):
        """Initialize a conditional node.

        Args:
            id: Optional node ID (generated if not provided)
            conditions: Dict mapping edge IDs to condition functions
            metadata: Optional metadata for the node
        """
        super().__init__(
            id=id or str(uuid4()),
            step=ChainStep(
                prompt=BasePrompt(
                    name="conditional_node",
                    description="Evaluates conditions to determine execution path",
                    system_prompt="",
                    user_prompt="",
                ),
            ),
            node_type=NodeType.CONDITIONAL,
            metadata=metadata or ChainMetadata(),
        )
        self.conditions = conditions or {}  # Use empty dict instead of None

    async def evaluate(self, input_value: Any) -> str:
        """Evaluate conditions and return the ID of the edge to follow.

        Args:
            input_value: The value to evaluate conditions against

        Returns:
            The ID of the edge to follow, or empty string if no condition matches
        """
        for edge_id, condition_func in self.conditions.items():
            if condition_func(input_value):
                return edge_id
        return ""  # Return empty string instead of None


class DynamicPromptNode(ChainNode):
    """A node that dynamically generates prompts based on tool results."""

    def __init__(
        self,
        prompt_template: BasePrompt,
        transform_func: Callable[[Dict[str, Any]], Dict[str, Any]],
    ):
        super().__init__(
            step=ChainStep(prompt=prompt_template),
            node_type=NodeType.AGENT,
        )
        self.transform_func = transform_func

    async def execute(
        self,
        agent_pool: AgentPool,
        context: ChainContext,
        dependency_results: Dict[str, Any],
        **kwargs: Any,
    ) -> Any:
        """Execute with dynamically generated prompt."""
        # Transform the dependency results into prompt variables
        variables = self.transform_func(dependency_results)

        # Update the prompt with the variables
        updated_prompt = BasePrompt(
            name=self.step.prompt.name,
            description=self.step.prompt.description,
            system_prompt=self.step.prompt.system_prompt,
            user_prompt=self.step.prompt.user_prompt.format(**variables),
            metadata=self.step.prompt.metadata,
            variables=self.step.prompt.variables,
        )

        # Execute the updated prompt
        return await agent_pool.execute_prompt(updated_prompt)


class ChainGraph(BaseGraph[ChainNode, ChainEdge]):
    """A graph-based representation of an LLM chain."""

    context: ChainContext = Field(default_factory=ChainContext)
    agent_pool: Optional[AgentPool] = None
    verify_acyclic: bool = Field(
        default=True, description="Whether to verify the graph is acyclic during initialization"
    )

    model_config = ConfigDict(validate_assignment=True, arbitrary_types_allowed=True)

    def __init__(self, **data: Any):
        """Initialize the chain graph and verify it is acyclic by default."""
        super().__init__(**data)
        # Get the verify_acyclic value from self after initialization
        should_verify = getattr(self, "verify_acyclic", True)
        # Verify the graph is acyclic by default if specified
        if should_verify and self.nodes and self.edges:
            self.verify_acyclic_graph()

    def verify_acyclic_graph(self) -> None:
        """Verify that the graph is acyclic (contains no cycles).

        Raises:
            ValueError: If a cycle is detected in the graph.
        """
        cycle = self.find_cycle()
        if cycle:
            cycle_str = " -> ".join(cycle)

            # Get node types for better debugging
            node_types = []
            for n_id in cycle:
                if n_id in self.nodes:
                    node_type = self.nodes[n_id].node_type.value
                    node_types.append(f"{n_id} ({node_type})")
                else:
                    node_types.append(n_id)

            node_type_str = " -> ".join(node_types)
            raise ValueError(f"Cycle detected in graph: {cycle_str}\n" f"Node types in cycle: {node_type_str}")

    def is_acyclic(self) -> bool:
        """Check if the graph is acyclic.

        Returns:
            bool: True if the graph is acyclic, False otherwise.
        """
        return self.find_cycle() is None

    def find_cycle(self) -> Optional[List[str]]:
        """Find a cycle in the graph if one exists.

        Returns:
            Optional[List[str]]: A list of node IDs forming a cycle, or None if no cycle exists.
        """
        # Use depth-first search to detect cycles
        visited = set()  # Nodes that have been fully processed
        path = []  # Current path being explored
        path_set = set()  # Set version of path for O(1) lookups
        cycle_found: List[Optional[List[str]]] = [None]  # Use a list to store the cycle

        def dfs(node_id: str) -> bool:
            """Depth-first search to detect cycles.

            Returns:
                bool: True if a cycle was found, False otherwise.
            """
            if node_id in path_set:
                # We've found a cycle
                cycle_start_idx = path.index(node_id)
                cycle_found[0] = path[cycle_start_idx:] + [node_id]
                return True

            if node_id in visited:
                return False

            visited.add(node_id)
            path.append(node_id)
            path_set.add(node_id)

            # Visit all neighbors
            for edge in self.edges:
                if edge.source_id == node_id:
                    if dfs(edge.target_id):
                        return True

            # Remove from current path
            path.pop()
            path_set.remove(node_id)
            return False

        # Start DFS from each node that hasn't been visited
        for node_id in self.nodes:
            if node_id not in visited:
                if dfs(node_id):
                    return cycle_found[0]

        return None

    def get_root_nodes(self) -> List[str]:
        """Get the IDs of all root nodes (nodes with no incoming edges)."""
        # Find all nodes that are targets of edges
        target_nodes = {edge.target_id for edge in self.edges}
        # Root nodes are those that are not targets of any edge
        root_nodes = [node_id for node_id in self.nodes.keys() if node_id not in target_nodes]
        return root_nodes

    async def execute(self, **kwargs: Any) -> Dict[str, Any]:
        """Execute the chain graph with conditional branching."""
        if not self.agent_pool:
            raise ValueError("AgentPool must be set before execution")

        results: Dict[str, Any] = {}
        execution_queue = list(self.get_root_nodes())  # Start with root nodes
        executed_nodes = set()

        while execution_queue:
            # Get next node to execute
            node_id = execution_queue.pop(0)
            node = self.nodes[node_id]

            # Skip if already executed
            if node_id in executed_nodes:
                continue

            # Check if dependencies are satisfied
            dependencies = self.get_node_dependencies(node_id)
            if not all(dep_id in executed_nodes for dep_id in dependencies):
                # Put back in queue if dependencies not met
                execution_queue.append(node_id)
                continue

            # Get results from dependencies
            dep_results = {dep_id: results[dep_id] for dep_id in dependencies}

            # Handle different node types
            if node.node_type == NodeType.CONDITIONAL:
                if isinstance(node, ConditionalNode):
                    # For conditional nodes, evaluate the condition
                    # Use the most recent dependency result as input
                    latest_dep_id = (
                        max(dependencies, key=lambda d: list(executed_nodes).index(d)) if dependencies else None
                    )
                    input_value = results[latest_dep_id] if latest_dep_id else None

                    # Evaluate conditions
                    chosen_edge_id = await node.evaluate(input_value)

                    # Find the target node of the chosen edge
                    next_node_id = None
                    for edge in self.edges:
                        if edge.source_id == node_id and edge.id == chosen_edge_id:
                            next_node_id = edge.target_id
                            break

                    # Add the next node to the execution queue
                    if next_node_id:
                        execution_queue.insert(0, next_node_id)

                    # Mark this node as executed
                    executed_nodes.add(node_id)
                    results[node_id] = chosen_edge_id  # Store the chosen path

                else:
                    raise ValueError(f"Node {node_id} is marked as CONDITIONAL but is not a ConditionalNode")

            else:
                # For regular nodes, execute normally
                result = await ChainExecutor.execute_with_retry(
                    node=node,
                    agent_pool=self.agent_pool,
                    context=self.context,
                    retry_strategy=node.step.retry_strategy,
                    dependency_results=dep_results,
                    **kwargs,
                )

                # Store result and mark as executed
                results[node_id] = result
                executed_nodes.add(node_id)

                # Add successor nodes to the queue
                for edge in self.edges:
                    if edge.source_id == node_id:
                        execution_queue.append(edge.target_id)

        return results


def create_tool_result_evaluator(
    tool_name: str, condition_func: Callable[[Any], bool]
) -> Callable[[Dict[str, Any]], bool]:
    """Create a function that evaluates a tool result.

    Args:
        tool_name: The name of the tool to evaluate
        condition_func: Function that takes the tool result and returns a boolean

    Returns:
        Function that takes a dependency_results dict and evaluates the condition
    """

    def evaluator(dependency_results: Dict[str, Any]) -> bool:
        for result in dependency_results.values():
            if isinstance(result, LLMResponse):
                # Check if the response has tool calls in its metadata
                tool_calls = result.metadata.get("tool_calls", [])
                for tool_call in tool_calls:
                    if tool_call.get("name") == tool_name:
                        return condition_func(tool_call.get("result"))
        return False

    return evaluator


def and_condition(*conditions: Callable[[Dict[str, Any]], bool]) -> Callable[[Dict[str, Any]], bool]:
    """Combine multiple conditions with AND logic."""
    return lambda results: all(condition(results) for condition in conditions)


def or_condition(*conditions: Callable[[Dict[str, Any]], bool]) -> Callable[[Dict[str, Any]], bool]:
    """Combine multiple conditions with OR logic."""
    return lambda results: any(condition(results) for condition in conditions)


def not_condition(condition: Callable[[Dict[str, Any]], bool]) -> Callable[[Dict[str, Any]], bool]:
    """Negate a condition."""
    return lambda results: not condition(results)
