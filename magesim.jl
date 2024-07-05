include("src/utils/include.jl")

import .Types: WorldState, AgentState, Logger, DummyNode, Config
import .World: create_world, world_step, stop_world
import .LogWriter: log
import .WorldRenderer: create_window, update_window!, close_window
import .AgentHandler: spawn_agents, step_agents!
import .ConfigLoader: load_configs

function main(args)

    if length(args) in [1, 2]
        configs = load_configs(args...)
    else
        throw(ArgumentError("Invalid number of arguments: $(length(args)). Please supply config name and optionally sweep config name as only arguments."))
    end

    # Due to constraints, headless-ness cannot vary across parameter sweep
    headless = configs[1].headless

    if !headless
        builder = create_window()
    end
    i = 1
    for cf in configs

        if !headless
            speedup = min(cf.speedup, 10.0)
        else
            speedup = cf.speedup
        end
        println("")
        print("------------------ Starting param_set $i / 495 ------------------")
        i += 1

        for run_n in 1:10 # HOW MANY RUNS
            println("")
            print("Starting run: ")
            println("$(run_n) / 10")
            world = create_world(cf)
            agents = spawn_agents(world, cf)
            ts = 1/speedup
            actual_speedup = speedup
            gtk_running = true
            if cf.do_log
                logger = Logger(cf, run_n)
                log_frequency = 1
            end

            for step in 1:cf.timeout
                t = @elapsed begin

                    step_agents!(agents, world, cf.multithreaded)
                    world_running, world, _ = world_step(world, agents)
                    
                    #println("Step: ", step)

                    if !headless
                        gtk_running = update_window!(world, agents, actual_speedup, builder)
                    end

                    if cf.do_log && step % log_frequency == 0 
                        for agent in agents
                            if agent.values.time_to_respond_log > 0
                                log(agent, logger, step)
                                agent.values.time_to_respond_log = 0
                            end
                            #log(world, logger, step)
                        end
                        log(world, logger, step)
                    end
                end

                if !headless && (world_running && gtk_running)
                    sleep(max(ts-t, 0))
                    actual_speedup = 1/max(t, ts)
                elseif !world_running
                    break
                end
            end

        
            stop_world()
        end
    end

    if !headless
        close_window(builder)
    end

end

main(ARGS)