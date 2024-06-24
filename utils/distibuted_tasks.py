def get_task_list(n_processors: int, current_processor_id: int, all_tasks: list):
    assert n_processors > 0
    assert current_processor_id > 0
    assert current_processor_id <= n_processors

    num_tasks_per_process = int(round(len(all_tasks) / n_processors))
    slice_start = (current_processor_id - 1) * num_tasks_per_process
    slice_end = slice_start + num_tasks_per_process
    if current_processor_id == n_processors:
        slice_end = len(all_tasks)
    return all_tasks[slice_start:slice_end]


if __name__ == "__main__":
    tasks = list(range(17))
    print(tasks)
    print(get_task_list(4, 1, tasks))
    print(get_task_list(4, 2, tasks))
    print(get_task_list(4, 3, tasks))
    print(get_task_list(4, 4, tasks))
