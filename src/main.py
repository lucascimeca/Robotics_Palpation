import numpy as np
from palpation import run_learning


def generate_action(current_state, idx, min_vec, max_vec, number_of_actions):
    if min_vec.shape[0] <= idx:
        return current_state
    else:
        if min_vec[idx] != 0 or max_vec[idx] != 0:
            if max_vec[idx] - min_vec[idx] == 0:
                actions = [max_vec[idx]]
            else:
            # actions = np.arange(min_vec[idx], max_vec[idx], (max_vec[idx]-min_vec[idx])/(number_of_actions + 1))[1:]
               actions = list(np.arange(min_vec[idx], max_vec[idx], (max_vec[idx] - min_vec[idx]) / (
                       number_of_actions - 1))) + [max_vec[idx]]
            if current_state is None:
                current_state = np.array([[0.0] * min_vec.shape[0]])

            state = current_state.copy()
            for i, act in enumerate(actions):
                if i == 0:
                    current_state[:, idx] = act
                else:
                    state_copy = state.copy()
                    state_copy[:, idx] = act
                    current_state = np.concatenate((current_state, state_copy), axis=0)
        return generate_action(current_state, idx+1, min_vec, max_vec, number_of_actions)


# Generate action profile, by using recursive function
def get_action_profile(eta_min, eta_max, A_min, A_max, number_of_actions=3):
    actions = generate_action(
        None,
        0,
        np.concatenate((eta_min, A_min), axis=0),
        np.concatenate((eta_max, A_max), axis=0),
        number_of_actions)
    action_profile = []
    for i in range(actions.shape[0]):
        action_profile += [(i, list(actions[i, :eta_min.shape[0]]), list(actions[i, eta_min.shape[0]:]))]
    return action_profile


if __name__ == "__main__":

    # x, y, z, rx, ry, rz -- > motion parameters to specify action: see paper
    # eta_min = np.array([0., 0., .5, 1., 1., 0.])
    # eta_max = np.array([0., 0., 2., 3., 3., 0.])
    # A_min = np.array([0., 0., .001, -np.deg2rad(10), -np.deg2rad(10), 0.])
    # A_max = np.array([0., 0., .005, np.deg2rad(10), np.deg2rad(10), 0.])  # 80 mm depth
    #
    # #### DEMO PARAMS
    # eta_min = np.array([0., 0., .5, 1., 1., 0.])
    # eta_max = np.array([0., 0., 2., 3., 3., 0.])
    # A_min = np.array([0., 0., .001, 0, 0, 0.])
    # A_max = np.array([0., 0., .005, np.deg2rad(10), np.deg2rad(10), 0.])  # 80 mm depth

    #### OFF PARAMS
    # x, y, z, rx, ry, rz -- > motion parameters to specify action: see paper
    # eta_min = np.array([0., 0., 0., 0., 0., 0.])
    # eta_max = np.array([0., 0., 1., 1., 1., 0.])
    # A_min = np.array([0., 0., 0, 0, 0, 0.])
    # A_max = np.array([0., 0., .005, np.deg2rad(7), -np.deg2rad(7), 0.])  # 80 mm depth

    eta_min = np.array([0., 0., 1., 1., 1., 0.])
    eta_max = np.array([0., 0., 1., 1., 1., 0.])
    A_min = np.array([0., 0., 0, 0, 0, 0.])
    A_max = np.array([0., 0., .005, np.deg2rad(7), -np.deg2rad(7), 0.])  # 80 mm depth

    # Generate 'number_of_actions' intervals for each of the min-max ranges given -- combinatorial!
    actions = get_action_profile(eta_min, eta_max, A_min, A_max, number_of_actions=6)

    training_environment = {
        '15mil': [(-.59148, -.00734), (-.60407, -.09483), (-.52041, -.10345), (-.51344, -.02892)],
        'na': [(-.54441, -.06520), (-.48423, -.07516), (-.60587, -.05676), (-.56625, -.12662)]
    }
    testing_environment = {
        '15mil': [(-.59148, -.00734), (-.60407, -.09483), (-.52041, -.10345), (-.51344, -.02892)],
        'na': [(-.54441, -.06520), (-.48423, -.07516), (-.60587, -.05676), (-.56625, -.12662)]
    }

    run_learning(
        time_interval=0,                    # how much does a step last (sec) -- 0 means as little as possible
        steps=0,                            # number of time steps -- 0 means infinite
        learning=True,                     # if learning then collect data,

        testing=False,
        resume_previous_experiment=False,   # if True the previous experiment if continued from where it was left off
        number_of_samples=5,               # how many times the robot palpates the same location, to learn
        sensing_resolution=10,               # how many skin samples to take during palpation contact time
        palpation_duration=5,               # duration of palpation contact time (sec),
        downtime_between_experiments=15,    # downtime between touches, to reset skin (sec),
        dimensionality_reduction=1,         # dimensionality reduction
        training_ratio=.7,                  # percentage of data for svm training - only matters if learning=False
        training_environment=training_environment,
        testing_environment=testing_environment,  # phantom environm ent - only relevant(used) for testing
        actions=actions[::-1],  # action list the robot must try
        verbose=False,
    )

