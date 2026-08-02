// Geometric planning for a rigid body in SE(3), following the OMPL tutorial.
// Shows both routes: with og::SimpleSetup and with the pieces wired up by hand.

#include <ompl/base/SpaceInformation.h>
#include <ompl/base/spaces/SE3StateSpace.h>
#include <ompl/geometric/SimpleSetup.h>
#include <ompl/geometric/planners/rrt/RRTConnect.h>
#include <ompl/config.h>

#include <iostream>

namespace ob = ompl::base;
namespace og = ompl::geometric;

// A spherical obstacle of radius 0.3 centred at the origin, so the planner has
// something to actually route around instead of returning a straight line.
bool isStateValid(const ob::State *state)
{
    const auto *se3state = state->as<ob::SE3StateSpace::StateType>();
    const auto *pos = se3state->as<ob::RealVectorStateSpace::StateType>(0);

    const double x = pos->values[0];
    const double y = pos->values[1];
    const double z = pos->values[2];

    return x * x + y * y + z * z > 0.3 * 0.3;
}

void planWithSimpleSetup()
{
    // construct the state space we are planning in
    auto space(std::make_shared<ob::SE3StateSpace>());

    ob::RealVectorBounds bounds(3);
    bounds.setLow(-1);
    bounds.setHigh(1);

    space->setBounds(bounds);

    og::SimpleSetup ss(space);

    ss.setStateValidityChecker([](const ob::State *state) { return isStateValid(state); });

    // The tutorial uses random start/goal states. We pin them to opposite
    // corners instead so the run is reproducible and the path has to detour
    // around the obstacle.
    ob::ScopedState<ob::SE3StateSpace> start(space);
    start->setXYZ(-0.9, -0.9, -0.9);
    start->rotation().setIdentity();

    ob::ScopedState<ob::SE3StateSpace> goal(space);
    goal->setXYZ(0.9, 0.9, 0.9);
    goal->rotation().setIdentity();

    ss.setStartAndGoalStates(start, goal);

    ob::PlannerStatus solved = ss.solve(1.0);

    if (solved)
    {
        std::cout << "Found solution:" << std::endl;
        // print the path to screen
        ss.simplifySolution();
        ss.getSolutionPath().print(std::cout);
    }
    else
        std::cout << "No solution found" << std::endl;
}

void plan()
{
    // construct the state space we are planning in
    auto space(std::make_shared<ob::SE3StateSpace>());

    ob::RealVectorBounds bounds(3);
    bounds.setLow(-1);
    bounds.setHigh(1);

    space->setBounds(bounds);

    auto si(std::make_shared<ob::SpaceInformation>(space));

    si->setStateValidityChecker(isStateValid);

    ob::ScopedState<ob::SE3StateSpace> start(space);
    start->setXYZ(-0.9, -0.9, -0.9);
    start->rotation().setIdentity();

    ob::ScopedState<ob::SE3StateSpace> goal(space);
    goal->setXYZ(0.9, 0.9, 0.9);
    goal->rotation().setIdentity();

    auto pdef(std::make_shared<ob::ProblemDefinition>(si));

    pdef->setStartAndGoalStates(start, goal);

    auto planner(std::make_shared<og::RRTConnect>(si));

    // Tell the planner which problem we are interested in solving
    planner->setProblemDefinition(pdef);

    // Make sure all the settings for the space and planner are in order. This
    // will also lead to the runtime computation of the state validity checking
    // resolution.
    planner->setup();

    ob::PlannerStatus solved = planner->ob::Planner::solve(1.0);

    if (solved)
    {
        // get the goal representation from the problem definition (not the same
        // as the goal state) and inquire about the found path
        ob::PathPtr path = pdef->getSolutionPath();
        std::cout << "Found solution:" << std::endl;

        // print the path to screen
        path->print(std::cout);
    }
    else
        std::cout << "No solution found" << std::endl;
}

int main()
{
    std::cout << "OMPL version: " << OMPL_VERSION << std::endl;

    std::cout << "\n=== planning with SimpleSetup ===" << std::endl;
    planWithSimpleSetup();

    std::cout << "\n=== planning without SimpleSetup ===" << std::endl;
    plan();

    return 0;
}
