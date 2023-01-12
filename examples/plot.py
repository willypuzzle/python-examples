import lib.models as m


def example1():
    my_feature = ([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
    my_label = ([5.0, 8.8, 9.6, 14.2, 18.8, 19.5, 21.4, 26.8, 28.9, 32.0, 33.8, 38.2])

    learning_rate = 0.01
    epochs = 10
    my_batch_size = 12

    my_model = m.build_model(learning_rate)

    trained_weight, trained_bias, epochs, rmse = m.train_model(my_model, my_feature,
                                                             my_label, epochs,
                                                             my_batch_size)
    m.plot_the_model(trained_weight, trained_bias, my_feature, my_label)
    m.plot_the_loss_curve(epochs, rmse)

