import random
import string
from model_factory import ModelFactory
from expression_factory import ExpressionFactory
from utils import setup_metamodel, save_model
from std_definitions import get_std_definitions
from resource_environment import get_resource_environment

class Edge:
    def __init__(self, number, source, target):
        self.number = number
        self.source = source
        self.target = target

    def __repr__(self):
        return f"Edge({self.source.number} -> {self.target.number})"

class Node:
    def __init__(self, number):
        self.number = number
        self.outgoing_edges = []
        self.ingoing_edges = []

    def add_outgoing_edge(self, edge):
        self.outgoing_edges.append(edge)

    def add_ingoing_edge(self, edge):
        self.ingoing_edges.append(edge)

    def __repr__(self):
        return f"Node({self.number})"

class DAG:
    def __init__(self, num_nodes, edge_prob):
        self.nodes = [Node(i) for i in range(num_nodes)]
        self.edges = []

        index = 0
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):  # ensure acyclicity
                if random.random() < edge_prob:
                    edge = Edge(index, self.nodes[i], self.nodes[j])
                    index += 1
                    self.nodes[i].add_outgoing_edge(edge)
                    self.nodes[j].add_ingoing_edge(edge)
                    self.edges.append(edge)

        # Find nodes with no ingoing edges (root nodes)
        self.root_nodes = [node for node in self.nodes if not node.ingoing_edges]
        self.boundary_edges = [Edge(i, None, self.root_nodes[i]) for i in range(len(self.root_nodes))]

class DAGModelGenerator:
    """Generator for PCM models, both minimal working examples and random models."""

    # TODO: add magic number definitions at the top so they can be globally configured easily for expirments (param_count, sig_count, provided_count... etc.)
    def __init__(self, seed=None):
        """Initialize the model generator.

        Args:
            seed: Optional random seed for reproducibility
        """
        # Set random seed if provided
        if seed is not None:
            random.seed(seed)

        # Initialize factories
        self.model_factory = ModelFactory()
        self.expr_factory = ExpressionFactory(self.model_factory.rset)
        self.std_defs = get_std_definitions(self.model_factory.rset)
        self.resource_env = get_resource_environment(self.model_factory.rset)

        # Store created elements for reference
        self.primitive_types = {}
        # same indices
        self.interfaces = []
        self.signatures = []
        self.roles = []
        #same indices
        self.components = []
        self.hardware_roles = []
        self.assembly_contexts = []
        # same indices
        self.resource_containers = []

        self.max_number_of_external_calls = 10
        # same indices
        self.boundary_interfaces = []
        self.boundary_roles = []


    def _random_name(self, prefix):
        """Generate a random name with a given prefix.

        Args:
            prefix: Prefix for the name

        Returns:
            A random name string
        """
        suffix = "".join(random.choices(string.ascii_uppercase, k=5))
        return f"{prefix}_{suffix}"

    def _create_primitive_types(self, repository):
        """Create primitive datatypes and add them to a repository.

        Args:
            repository: Repository to add types to

        Returns:
            Dictionary of created types
        """
        # Use the same names as in std_definitions.py
        type_enums = {
            "Integer": self.model_factory.PCM.PrimitiveTypeEnum.INT,
            "String": self.model_factory.PCM.PrimitiveTypeEnum.STRING,
            "Boolean": self.model_factory.PCM.PrimitiveTypeEnum.BOOL,
            "Double": self.model_factory.PCM.PrimitiveTypeEnum.DOUBLE,
        }

        for name, enum_val in type_enums.items():
            datatype = self.model_factory.create_primitive_datatype(name, enum_val)
            repository.contents.append(datatype)
            self.primitive_types[name] = datatype

        return self.primitive_types

    def _create_random_signature(self, return_type=None):
        """Create a random operation signature.

        Args:
            interface: Interface to add the signature to
            return_type: Optional return type

        Returns:
            Created signature
        """
        # Choose random name and return type
        name = self._random_name("operation")
        if return_type is None:
            return_type = random.choice(list(self.primitive_types.values()))

        # Create signature
        signature = self.model_factory.create_operation_signature(name, return_type)

        # Add random parameters (0-3)
        param_count = random.randint(0, 3)
        for i in range(param_count):
            param_type = random.choice(list(self.primitive_types.values()))
            param = self.model_factory.create_parameter(f"param{i}", param_type)
            signature.parameters.append(param)

        return signature

    def _create_parameter_random_Integer(self, reference = None):
        return self.model_factory.create_parameter_specification(
            reference = reference,
            specification=self.expr_factory.create_int_literal(
                random.randint(1, 100)
            )
        )
    
    def _create_parameter_random_String(self, reference = None):
        length = random.randint(5, 15)
        random_string = "".join(
            random.choices(
                string.ascii_letters + string.digits, k=length
            )
        )
        return self.model_factory.create_parameter_specification(
            reference = reference,
            specification=self.expr_factory.create_string_literal(
                random_string
            )
        )

    def _create_parameter_random_Boolean(self, reference = None):
        return self.model_factory.create_parameter_specification(
            reference = reference,
            specification=self.expr_factory.create_bool_literal(
                random.choice([True, False])
            )
        )
    
    def _create_parameter_random_Double(self, reference = None):
        return self.model_factory.create_parameter_specification(
            reference = reference,
            specification=self.expr_factory.create_double_literal(
                round(random.uniform(1.0, 100.0), 2)
            )
        )
    
    def _create_parameter(self, param, reference = None):
        if param.type == self.primitive_types["Integer"]:
            return self._create_parameter_random_Integer(reference)
        elif param.type == self.primitive_types["String"]:
            return self._create_parameter_random_String(reference)
        elif param.type == self.primitive_types["Boolean"]:
            return self._create_parameter_random_Boolean(reference)
        # FIXME: Current workaround because of type issue with CPU and HDD
        else:
            return self._create_parameter_random_Double(reference)



    def generate_repository(self):
        """Generate a random repository with interfaces and components.

        Args:
            dag: directed acyclic graph resembling the system to generate

        Returns:
            Generated repository
        """
        # Create repository
        repository = self.model_factory.create_repository(
            self._random_name("repository")
        )

        # Create primitive types
        self._create_primitive_types(repository)

        # Create components
        self.components = [self.model_factory.create_component(
                self._random_name("component")
            ) for i in range(len(self.dag.nodes))]

        [repository.contents.append(component) for component in self.components]

        # Create interfaces, one per edge
        # FIXME: It should be possible for interfaces to be shared between edges
        self.interfaces = [self.model_factory.create_domain_interface(
                self._random_name("interface")
            ) for i in range(len(self.dag.edges))]

        # Add a signature to each interface
        [interface.contents.append(self._create_random_signature()) for interface in self.interfaces]

        [repository.contents.append(interface) for interface in self.interfaces]

        for interface, edge in zip(self.interfaces, self.dag.edges):
            providing_component = self.components[edge.target.number]
            provided_role = self.model_factory.create_provided_role(
                self._random_name("provided"), interface
            )
            providing_component.contents.append(provided_role)
            requiring_component = self.components[edge.source.number]
            required_role = self.model_factory.create_required_role(
                self._random_name("required"), interface
            )
            requiring_component.contents.append(required_role)
            self.roles.append([provided_role, required_role])

        for component in self.components:
            cpu_role = self.model_factory.create_required_role(
                "cpu", self.std_defs.get_cpu_interface()
            )
            component.contents.append(cpu_role)
            hdd_role = self.model_factory.create_required_role(
                "hdd", self.std_defs.get_hdd_interface()
            )
            component.contents.append(hdd_role)
            self.hardware_roles.append([cpu_role, hdd_role])

        for node in self.dag.nodes:
            component = self.components[node.number]
            required_roles = [
                content
                for content in component.contents 
                if isinstance(content, self.model_factory.PCM.InterfaceRequiredRole)
                ]
            for edge in node.ingoing_edges:
                for signature in self.roles[edge.number][0].type.contents:
                    provided_role = self.roles[edge.number][0]
                    seff = self.model_factory.create_seff(provided_role, signature)
                    number_of_external_calls = random.randint(0, min(self.max_number_of_external_calls, len(required_roles)))
                    roles_for_external_calls = random.sample(required_roles, number_of_external_calls)
                    for required_role in roles_for_external_calls:
                        required_signature = random.choice(required_role.type.contents)
                        parameters = required_signature.parameters

                        external_call = self.model_factory.create_seff_call_action(
                            required_role, required_signature
                        )

                        for parameter in parameters:
                            external_call.parameters.append(self._create_parameter(parameter))

                        seff.contents.append(external_call)
                    component.contents.append(seff)

        # Boundary interfaces
        self.boundary_interfaces = [self.model_factory.create_domain_interface(
                self._random_name("interface")
            ) for i in range(len(self.dag.boundary_edges))]
        [interface.contents.append(self._create_random_signature()) for interface in self.boundary_interfaces]
        [repository.contents.append(interface) for interface in self.boundary_interfaces]

        for boundary_interface, boundary_edge in zip(self.boundary_interfaces, self.dag.boundary_edges):
            providing_component = self.components[boundary_edge.target.number]
            provided_role = self.model_factory.create_provided_role(
                self._random_name("provided"), boundary_interface
            )
            providing_component.contents.append(provided_role)
            self.boundary_roles.append(provided_role)

        for boundary_edge in self.dag.boundary_edges:
            component = self.components[boundary_edge.target.number]
            required_roles = [
                content
                for content in component.contents 
                if isinstance(content, self.model_factory.PCM.InterfaceRequiredRole)
                ]
            for signature in self.boundary_roles[boundary_edge.number].type.contents:
                    provided_role = self.boundary_roles[boundary_edge.number]
                    seff = self.model_factory.create_seff(provided_role, signature)
                    roles_for_external_calls = random.sample(required_roles, random.randint(1, min(self.max_number_of_external_calls, len(required_roles))))
                    for required_role in roles_for_external_calls:
                        required_signature = random.choice(required_role.type.contents)
                        parameters = required_signature.parameters

                        external_call = self.model_factory.create_seff_call_action(
                            required_role, required_signature
                        )

                        for parameter in parameters:
                            external_call.parameters.append(self._create_parameter(parameter))

                        seff.contents.append(external_call)
                    component.contents.append(seff)

        return repository

    def generate_system(self):
        """Generate a random system with assembly contexts and connectors.

        Args:
            repository: Repository with components and interfaces
            Sebastian: Ähm?!

        Returns:
            Generated system
        """
        # Create system
        system = self.model_factory.create_system(self._random_name("system"))

        for component in self.components:
            assembly = self.model_factory.create_assembly_context(
                self._random_name("assembly"), component
            )
            system.contents.append(assembly)
            self.assembly_contexts.append(assembly)

        for edge in self.dag.edges:
            from_context=self.assembly_contexts[edge.target.number]
            to_context=self.assembly_contexts[edge.source.number]
            requiring_roles = [
                content
                for content in from_context.component.contents 
                if isinstance(content, self.model_factory.PCM.InterfaceRequiredRole)
                ]
            requiring_role=next((required_role for required_role in requiring_roles if required_role.type == self.interfaces[edge.number]), None)
            connector = self.model_factory.create_connector(
                to_context=to_context,
                from_context=from_context,
                requiring_role=requiring_role,
            )

            system.contents.append(connector)

        for boundary_edge in self.dag.boundary_edges:
            system_role = self.model_factory.create_system_provided_role(
                self._random_name("system_provided"), 
                self.boundary_interfaces[boundary_edge.number], 
                self.assembly_contexts[boundary_edge.target.number]
            )
            system.contents.append(system_role)
        return system

    def generate_allocation(self, system):
        """Generate an allocation of a system to resource containers.

        Args:
            system: System to allocate

        Returns:
            Generated allocation
        """
        # Create allocation
        allocation = self.model_factory.create_allocation(
            self._random_name("allocation")
        )

        # Get the resource containers from the resource environment
        containers = self.resource_env.get_resource_containers()

        if not containers or not system:
            return allocation

        # Group assemblies for allocation to containers
        assemblies = [
            content
            for content in system.contents
            if hasattr(content, "eClass") and content.eClass.name == "AssemblyContext"
        ]

        if not assemblies:
            return allocation

        # Distribute assemblies across resource containers randomly
        num_groups = min(len(containers), len(assemblies))
        # Create at least one group, but not more than we have containers or assemblies
        num_groups = max(1, random.randint(1, num_groups))

        # Split assemblies into groups
        assembly_groups = []
        assemblies_copy = assemblies.copy()
        random.shuffle(assemblies_copy)

        # Distribute assemblies evenly across groups
        group_size = len(assemblies_copy) // num_groups
        remainder = len(assemblies_copy) % num_groups

        start = 0
        for i in range(num_groups):
            size = group_size + (1 if i < remainder else 0)
            end = start + size
            if size > 0:
                assembly_groups.append(assemblies_copy[start:end])
            start = end

        # Create allocation contexts
        for i, group in enumerate(assembly_groups):
            if i < len(containers) and group:
                container = containers[i]
                alloc_ctx = self.model_factory.create_allocation_context(
                    self._random_name("alloc"), group, container
                )
                allocation.contents.append(alloc_ctx)

        return allocation

    def generate_usage_model(self, system):
        """Generate a usage model for a system.

        Args:
            system: System the usage model is for

        Returns:
            Generated usage model
        """
        # Create usage model
        usage = self.model_factory.create_usage_model(self._random_name("usage"))

        # Find system provided roles that can be called
        system_provided_roles = [
            content
            for content in system.contents
            if isinstance(content, self.model_factory.PCM.SystemProvidedRole)
        ]

        if not system_provided_roles:
            return usage

        # Create a usage scenario
        scenario = self.model_factory.create_usage_scenario(
            self._random_name("scenario")
        )

        # Create workload (randomly choose between open and closed workload)
        if random.choice([True, False]):
            # Open workload with exponential distribution
            rate = random.uniform(0.01, 0.1)
            # Create a simple double literal directly
            inter_arrival_time = self.expr_factory.create_double_literal(rate)
            workload = self.model_factory.create_open_workload(inter_arrival_time)
        else:
            # Closed workload
            num_users = random.randint(1, 20)
            think_time = self.expr_factory.create_double_literal(
                random.uniform(0.5, 5.0)
            )
            workload = self.model_factory.create_closed_workload(num_users, think_time)

        scenario.workload = workload

        # Create entry level system calls to random system provided roles
        call_count = random.randint(1, 10)
        for _ in range(call_count):
            # Choose random role
            role = random.choice(system_provided_roles)

            # Find a signature from the interface
            if role.type and role.type.contents:
                signatures = [
                    sig
                    for sig in role.type.contents
                    if isinstance(sig, self.model_factory.PCM.Signature)
                ]

                if signatures:
                    signature = random.choice(signatures)

                    # Create parameter specifications for parameters if needed
                    params = []
                    for param in signature.parameters:
                        # Create random parameter values based on parameter type
                        namespace_reference = self.expr_factory.create_namespace_reference(param.name, self.expr_factory.create_variable_reference("VALUE"))
                        absolute_reference = self.model_factory.create_absolute_reference(namespace_reference)
                        params.append(self._create_parameter(param, absolute_reference))

                    # Create entry level system call
                    call = self.model_factory.create_entry_level_system_call(
                        role, signature, params
                    )
                    scenario.contents.append(call)

        # Add scenario to usage model
        usage.contents.append(scenario)

        return usage

    def generate_complete_model(self, output_file="generated/generated.xml", nodes=10, edge_probability=0.3):
        """Generate a complete PCM model with all elements.

        Args:
            model_name: Base name for the model and output files

        Returns:
            Tuple of (model, model_resource)
        """
        # Create model
        model = self.model_factory.create_model()

        # Add standard definitions and resource environment
        self.std_defs.add_to_model(model)
        # Get resource environment from the singleton pattern
        self.resource_env.add_to_model(model)

        self.dag = DAG(nodes, edge_probability)

        # Generate all model elements
        repository = self.generate_repository()
        system = self.generate_system()
        allocation = self.generate_allocation(system)
        usage = self.generate_usage_model(system)

        # Add elements to model
        model.fragments.extend([repository, system, allocation, usage])

        # Save model
        model_resource = save_model(model, output_file, self.model_factory.rset)

        return model, model_resource