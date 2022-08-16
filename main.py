from data_acquisition.main import load_trajectories_and_returns_from_file, load_trajectories_and_returns_from_env, process_trajectories
from training.model import get_reward_model
from training.main import train_model

if __name__ == '__main__':

    T = 50
    memory_window_length = 2
    # Create a multi-input model, with each input is a fixed window of the trajectory
    num_windows = T - memory_window_length + 1
    model = get_reward_model(num_windows, memory_window_length)

    # Load the trajectories and their corresponding returns
    trajectories, returns = load_trajectories_and_returns_from_env('ToyCenteredArray', 200_000, T)
    # trajectories, returns = load_trajectories_and_returns_from_file('data_acquisition/data/toy_centered_array', 1_000_000, T)
    x_train, y_train = process_trajectories(trajectories, returns, memory_window_length, shuffle=True)

    train_model(model, x_train, y_train)


