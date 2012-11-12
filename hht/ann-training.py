from pyfann import libfann


connection_rate = 1
learning_rate = 0.7
num_input = 5
num_hidden = 4
num_output = 1

desired_error = 1
max_iterations = 100000
iterations_between_reports = 1000

ann = libfann.neural_net()
#ann.create_shortcut_array()
#ann.create_standard_array((4, 2, 4, 4, 1))
ann.create_sparse_array(connection_rate, (num_input, num_hidden, num_output))
#ann.set_learning_rate(learning_rate)
ann.set_activation_function_output(libfann.SIGMOID_SYMMETRIC_STEPWISE)
ann.set_training_algorithm(libfann.TRAIN_RPROP)
ann.set_train_error_function(libfann.ERRORFUNC_TANH)
ann.train_on_file("testfile.data", max_iterations, iterations_between_reports, desired_error)

ann.save("testfile.net")