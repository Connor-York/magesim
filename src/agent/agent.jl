module Agent

import ..Types: AgentState, WorldState, Position, AbstractAction, WaitAction, ScanAction, DelayAction, MoveToAction, StepTowardsAction, StringMessage, ArrivedAtNodeMessage, RecruitMessage, RecruitResponse, MissionComplete, ObservedNodeMessage
import ..AgentDynamics: calculate_next_position
import ..Utils: get_neighbours, pos_distance, get_distances, check_if_observed
using DataStructures

# RANDOMISE NODE response time? Agents want to compete for auctions? PLASTICITY?
"""
    agent_step!(agent::AgentState, world::WorldState)

Select and perform an action and update agent position and stat accordingly
"""

function agent_step!(agent::AgentState, world::WorldState, blocked_pos::Array{Position, 1})

    # Wait if no other action found
    if isempty(agent.action_queue)
        enqueue!(agent.action_queue, WaitAction())
    end

    # Do action from queue

    action = first(agent.action_queue)

    if action isa WaitAction
        # Do nothing for one timestep and then decrement wait duration
        new_pos = agent.position
        new_graph_pos = agent.graph_position
        action.duration -= 1
        if action.duration <= 0
            action_done = true
        else
            action_done = false
        end
    elseif action isa ScanAction
        new_pos = agent.position
        new_graph_pos = agent.graph_position
        # if agent.id == 1
        #     println("Agent $(agent.id) is scanning, time = $(action.duration)")
        # end
        action.duration -= 1
        enqueue!(agent.outbox, ArrivedAtNodeMessage(agent, nothing, (agent.graph_position, agent.graph_position))) #keep informing everyone they are here
        if action.duration <= 0
            if agent.values.free[1] == false 
                if agent.values.free[2] != agent.id # if smart node
                    enqueue!(agent.outbox, MissionComplete(agent, [agent.values.free[2]]))
                end
                agent.values.free = (true, agent.values.free[2])
            end
            #println("Agent finished scan 2")
            action_done = true
            agent.values.reward_log += 1
            #println("Agent $(agent.id) getting reward from node $(agent.graph_position), now at reward $(agent.values.reward_log)")
        else
            action_done = false
        end
    elseif action isa DelayAction
        new_pos = agent.position
        new_graph_pos = agent.graph_position
        action.duration -= 1
        if action.duration <= 0
            enqueue!(agent.outbox, RecruitMessage(agent, nothing, false)) # delay done send another order
            #println("SN $(agent.id) finished delay after rejection/ no response sending another recruitment")
            action_done = true
        else
            action_done = false
        end
    elseif action isa MoveToAction
        # Move towards target and do not pop action from queue until target reached
        new_pos, new_graph_pos, action_done = calculate_next_position(agent, action.target, world, blocked_pos)
    elseif action isa StepTowardsAction
        # Take one step towards target 
        new_pos, new_graph_pos, _ = calculate_next_position(agent, action.target, world, blocked_pos)
        action_done = true
    else
        error("Error: no behaviour found for action of type $(nameof(typeof(action)))")
    end

    agent.position = new_pos
    agent.graph_position = new_graph_pos

    if action_done
        dequeue!(agent.action_queue)
    end

end

"""
    observe_world!(agent::AgentState, world::WorldState)

Extract an agent's observation from the true world state and update the agent's belief of the
world state, and generate messages to send to other agents
"""
function observe_world!(agent::AgentState, world::WorldState, anomaly_duration::Int64)

    agent.world_state_belief = world


    
    if !agent.values.stationarity #regular agents update idleness
        agent.values.idleness_log = [i + 1.0 for i in agent.values.idleness_log]
        if agent.graph_position isa Int64 && agent.graph_position <= world.n_nodes # if you want moving agents to not update idleness change this
            agent.values.idleness_log[agent.graph_position] = 0.0
        end
    end



    if agent.values.stationarity && !agent.values.anomalous[1] # if you're already anomalous youve checked already, maybe redundant
        if agent.world_state_belief.nodes[agent.graph_position].values.anomalous[1] # if current node is anomalous
            if check_if_observed(agent.graph_position, agent) == false
                agent.values.anomalous = agent.world_state_belief.nodes[agent.graph_position].values.anomalous
                push!(agent.values.observed_anomalies, (agent.graph_position, world.time))
                enqueue!(agent.outbox, ObservedNodeMessage(agent, nothing, (agent.graph_position, world.time)))
                enqueue!(agent.outbox, RecruitMessage(agent, nothing, false)) # send out recruitment message
            end
        end
    elseif !agent.values.stationarity && agent.values.free[1] # busy agents dont check
        if agent.graph_position isa Int64
            if agent.world_state_belief.nodes[agent.graph_position].values.anomalous[1] && !agent.world_state_belief.nodes[agent.graph_position].values.smart# if current node is anomalous
                if check_if_observed(agent.graph_position, agent) == false
                    # println("Agent $(agent.id) checking node $(agent.graph_position), observed: $(check_if_observed(agent.graph_position, agent))")
                    # println("Agent $(agent.id) observed anomalies: $(agent.values.observed_anomalies)")
                    agent.values.free = (false, agent.id)
                    push!(agent.values.observed_anomalies, (agent.graph_position, world.time))
                    enqueue!(agent.outbox, ObservedNodeMessage(agent, nothing, (agent.graph_position, world.time)))
                    enqueue!(agent.action_queue, ScanAction())
                else
                    # println("NODE HAS BEEN OBSERVED")
                end
            end
        end
    end

    #println("Agent $(agent.id) observed anomalies: $(agent.values.observed_anomalies)")
    for node in agent.values.observed_anomalies # clear timed out anomalies
        if agent.world_state_belief.time - node[2] >= anomaly_duration
            #println("Agent $(agent.id) clearing node $(node[1])")
            filter!(x -> x != node, agent.values.observed_anomalies)
        end
    end

    # enqueue!(agent.outbox, StringMessage(agent, nothing, string(agent.id)))
end


"""
    make_decisions!(agent::AgentState)

Read messages and modify agent's action queue based on received messages, world state belief, and 
internal values
"""
function make_decisions!(agent::AgentState)
    patrol_method::String = "SEBS" #CGG or SEBS

    while !isempty(agent.inbox)
        message = dequeue!(agent.inbox)

        if message isa ArrivedAtNodeMessage #SEBS
            agent.values.idleness_log[message.message[1]] = 0.0
            agent.values.intention_log[message.source] = message.message[2]
        end

        if message isa ObservedNodeMessage
            if check_if_observed(message.message[1], agent) == false
                push!(agent.values.observed_anomalies, message.message)
            end
        end

        if !agent.values.stationarity # agents

            if message isa RecruitMessage

                if message.order == false
                    enqueue!(agent.outbox, RecruitResponse(agent, [message.source], false)) # send out response, not rejection
                elseif message.order == true
                    if agent.values.free[1]
                        r = 1.0 # rand()
                        if r > agent.values.stubborn
                            push!(agent.values.recruitment_bids, (0.0, message)) # send out response, acceptance
                        else
                            enqueue!(agent.outbox, RecruitResponse(agent, [message.source], true)) # send out response, rejection
                        end
                    elseif !agent.values.free[1]
                        enqueue!(agent.outbox, RecruitResponse(agent, [message.source], true)) # send out response, rejection
                    end
                end

            end

        elseif agent.values.stationarity # smart_nodes
        
            if message isa MissionComplete
                agent.values.time_to_respond_log = agent.world_state_belief.time - agent.values.anomalous[2]
                agent.values.anomalous = (false, 0)
            end

            if message isa RecruitResponse

                if message.rejection == false
                    stubborness = message.stubborness
                    distance = get_distances(message.agent_graph_position, message.agent_position, message.world_state_belief)[agent.graph_position]
                    value = distance * stubborness
                    push!(agent.values.recruitment_bids, (value, message))
                elseif message.rejection == true
                    enqueue!(agent.action_queue, DelayAction())
                end

            end

        end       
    end

    #   messages parsed, choose from responses (recruitment for smart_nodes, orders for agents)
    if length(agent.values.recruitment_bids) > 0

        if !agent.values.stationarity && agent.values.free[1] # agents (free)

            if length(agent.values.recruitment_bids) == 1
                empty!(agent.action_queue)
                enqueue!(agent.action_queue, MoveToAction(agent.values.recruitment_bids[1][2].smart_node_position))
                agent.values.free = (false, agent.values.recruitment_bids[1][2].source)

            elseif length(agent.values.recruitment_bids) > 1 

                # if received multiple orders in this timestep, choose one at random and reject the others
                r = rand(1:length(agent.values.recruitment_bids))
                for (index, bid) in enumerate(agent.values.recruitment_bids)
                    if index == r
                        empty!(agent.action_queue)
                        enqueue!(agent.action_queue, MoveToAction(bid[2].smart_node_position))
                        agent.values.free = (false, bid[2].source)
                    else
                        enqueue!(agent.outbox, RecruitResponse(agent, [bid[2].source], true))

                    end
                end

            end

            empty!(agent.values.recruitment_bids)

        elseif agent.values.stationarity # smart_nodes

            if length(agent.values.recruitment_bids) == 1
                if agent.values.recruitment_bids[1][2].free[1] == true
                    enqueue!(agent.outbox, RecruitMessage(agent, [agent.values.recruitment_bids[1][2].source], true))
                elseif agent.values.recruitment_bids[1][2].free[1] == false
                    enqueue!(agent.action_queue, DelayAction())
                end
            else
                sort!(agent.values.recruitment_bids, by=x->x[1])
                chosen = false
                for i in agent.values.recruitment_bids
                    if i[2].free[1] == true
                        enqueue!(agent.outbox, RecruitMessage(agent, [i[2].source], true))
                        chosen = true
                        break
                    end
                end

                if chosen == false
                    enqueue!(agent.action_queue, DelayAction())
                end
            end

            empty!(agent.values.recruitment_bids)

        end

    end

    if !isnothing(agent.world_state_belief) #Give next action


        if isempty(agent.action_queue) #if action queue is empty 
            if agent.graph_position isa Int64 # if at a node 



                if !agent.values.stationarity && !agent.values.free[1] #if not free
                    #r = rand(1:3)
                    #delays[]
                    enqueue!(agent.action_queue, ScanAction(100))
                end

                #  GO TO NEXT NODE ------------------------- 
                if !agent.values.stationarity && agent.values.free[1] #if free
                    if patrol_method == "CGG"
                        if agent.graph_position == 25
                            next_node = 1
                            enqueue!(agent.action_queue, MoveToAction(next_node))
                        else 
                            next_node = agent.graph_position + 1
                            enqueue!(agent.action_queue, MoveToAction(next_node))
                        end


                    elseif patrol_method == "SEBS"
                        
                        neighbours = get_neighbours(agent.graph_position, agent.world_state_belief, true)

                        if length(neighbours) == 1
                            enqueue!(agent.action_queue, MoveToAction(neighbours[1]))
                        elseif !isa(agent.graph_position, Int64)
                            # Catch the potential problem of an agent needing a new action
                            # while midway between two nodes (not covered by algo) - 
                            # solution to this is just to pick one
                            enqueue!(agent.action_queue, MoveToAction(neighbours[1]))
                        else
                            # Do SEBS
                            gains = map(n -> calculate_gain(n, agent), neighbours)
                            posteriors = map(g -> calculate_posterior(g, agent), gains)
                            n_intentions::Array{Int64, 1} = zeros(agent.world_state_belief.n_nodes)
                            for i in agent.values.intention_log
                                if i != 0 n_intentions[i] += 1 end
                            end
                            intention_weights = map(n -> calculate_intention_weight(n, agent), n_intentions)
                            final_posteriors = [posteriors[i] * intention_weights[neighbours[i]] for i in 1:length(posteriors)]
                            target = neighbours[argmax(final_posteriors)]
                            enqueue!(agent.action_queue, MoveToAction(target))
                            enqueue!(agent.outbox, ArrivedAtNodeMessage(agent, nothing, (agent.graph_position, target)))
                        end
                    end
                end
            end
        end

        #rand(1:agent.world_state_belief.n_nodes)
    end
end


function calculate_gain(node::Int64, agent::AgentState)
    distance = agent.world_state_belief.paths.dists[agent.graph_position, node]
    return agent.values.idleness_log[node] / distance
end

function calculate_posterior(gain::Float64, agent::AgentState)
    g1 = 0.1
    g2 = 100

    if gain >= g2
        return 1.0
    else
        return g1 * 2.7183^((gain/g2) * log(1/g1))
    end
end

function calculate_intention_weight(n_intentions::Int64, agent::AgentState)

    n_agents = agent.values.n_agents_belief
    return 2^(n_agents - n_intentions)/(2^n_agents - 1)
end

end