## MAGESim

## Interface with PettingZoo ParallelEnv API
### Limitations
When using the provided wrapper to create a PettingZoo ParallelEnv object that wraps the simulator code, there are several restrictions that are not otherwise present that must be observed.
* `NodeValues` can only have fields of types `Int`, `Float`, `Bool`, `String`, or 1-d `Array` of these types. Default values of `Int` or `Array{Int}` must refer to the maximum possible value of the field, and no negative values are allowed. As such, it is recommended to only use these types to represent enums, and to use `Float` or `Array{Float}` to represent values that may be negative or may have unknown upper bounds. Default values of `String` or `Array{String}` must be strings of the integer values of the maximum allowable length (in characters) of the strings.
* Currently, all agents must have the same observation space (but may make different observations within that space).
* The only action permitted is `StepTowardsAction`, as this is the most straightforwardly compatible with the standard RL flow of observe -> select action -> step environment -> observe.
* Manual control of speedup is disabled, the simulator will run as fast as possible.
* Multi-threading of agents is currently not supported.
### Action spaces
Currently, the only action compatible with the interface is the `StepTowardsAction`. The corresponding action space is a `Discrete` space with a size equal to the number of nodes in the map graph.
### Observation spaces
Currently, the observation space of all agents is that agent's belief of the world state, existing in `Agent.world_state_belief`. This consists of the map, the agent's location, and all node values. The map is represented as a `Graph` space observing node positions, edges, and optional edge weights (custom edge weights (ie. other than Cartesian distance) not currently implemented). The agent's location is represented as a `Box` in R2. The representation of node values varies depending on the fields of the `NodeValues` struct, but will be a composite space of n instances of the space generated in the wrapper code to represent an individual instance of `NodeValues`.