module LogWriter

import ..Types: Logger, WorldState, AgentState, Node
using Graphs, SimpleWeightedGraphs
using Dates

"""
    log(target::AgentState, logger::Logger, timestep::Int)

Log AgentState data
"""
function log(target::AgentState, logger::Logger, timestep::Int)
    fpath = string(logger.log_directory, "$(target.id)/$(logger.run_n)_Rewards.csv") 

    if !isdir(string(logger.log_directory, "$(target.id)/"))
        Base.Filesystem.mkpath(string(logger.log_directory, "$(target.id)/"))
    end

    if !isfile(fpath)
        header = "reward, timestep"
        open(fpath, "w") do file
            write(file, header)
            write(file,"\n")
        end
    end

    csv_line = "$(target.values.cumulative_reward), $(string(timestep))"

    open(fpath, "a") do file
        write(file, csv_line)
        write(file,"\n")
    end
end

"""
    log(target::Array{AgentState, 1}, logger::Logger, timestep::Int)

Log multiple instances of AgentState data
"""
function log(target::Array{AgentState, 1}, logger::Logger, timestep::Int)
   
    header_contents = ["x$n,y$n" for n in [1:1:length(target)...]]
    positions = vcat([[agent.position.x, agent.position.y] for agent in target]...)


    fpath = string(logger.log_directory, "agent_positions.csv") 

    if !isfile(fpath)
        header = make_line("timestep", header_contents)
        open(fpath, "w") do file
            write(file, header)
            write(file,"\n")
        end
    end

    csv_line = make_line(timestep, string.(positions))
    open(fpath, "a") do file
        write(file, csv_line)
        write(file,"\n")
    end
end

"""
    log(target::Node, logger::Logger, timestep::Int)

Log Node data
"""
function log(target::Node, logger::Logger, timestep:: Int)
    fpath = string(logger.log_directory, "node_", string(target.id), ".csv")

    if !isfile(fpath)
        header = "timestep, value"
        open(fpath, "w") do file
            write(file, header)
            write(file,"\n")
        end
    end

    csv_line = string(string(timestep, target.values.value_string))

    open(fpath, "a") do file
        write(file, csv_line)
        write(file,"\n")
    end
end

"""
    log(target::WorldState, logger::Logger, timestep::Int)

Log WorldState data
"""
function log(target::WorldState, logger::Logger, timestep::Int)

    header_contents = ["node_$n" for n in [1:1:target.n_nodes...]]
    idlenesses = [node.values.idleness for node in target.nodes if node isa Node]

    fpath = string(logger.log_directory, "world.csv") 

    if !isfile(fpath)
        header = make_line("timestep", string.(header_contents))
        open(fpath, "w") do file
            write(file, header)
            write(file,"\n")
        end
    end

    csv_line = make_line(timestep, string.(idlenesses))

    open(fpath, "a") do file
        write(file, csv_line)
        write(file,"\n")
    end
end

"""
    Utility functions for formatting
"""
function make_line(timestep::Int, contents::Array{String, 1})
    return join(vcat(string(timestep), contents), ',')
end
function make_line(timestep::String, contents::Array{String, 1})
    return join(vcat(timestep, contents), ',')
end

end