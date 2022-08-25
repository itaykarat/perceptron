import PySimpleGUI as sg
from Graphics.Plotter import plotter
from Algo.perceptron_algo import perceptron_algo
from Data_creation.synthetic_dataset_creation import synthetic_data


class application:
    def __init__(self):
        self.theme = 'DarkAmber'

    def RUN(self):
        sg.theme(self.theme)  # Add a touch of color
        layout = [[sg.Button('Display synthetic data'),sg.Slider(orientation ='horizontal', key='learning_rate_slider', range=(0.0,1.0), resolution=.01)],
                  [sg.Button('Find linear seperator'),sg.Slider(orientation ='horizontal', key='Epoches_slider', range=(0,100))],
                  [sg.Button('Cancel')]]

        # Create the Window
        window = sg.Window('Window Title', layout)
        data = synthetic_data().create_synthetic_data_set()  # creating synthetic data
        # Event Loop to process "events" and get the "values" of the inputs
        while True:
            event, values = window.read()
            if event == 'Display synthetic data':
                plotter().plot_data(data)  # plot the synthetic data

            if event == 'Find linear seperator':
                theta, n_miss_list = perceptron_algo().perceptron(data=data, lr=values['learning_rate_slider'], epochs=int(values['Epoches_slider']))
                plotter().plot_decision_boundary(data=data, theta=theta)

            if event == sg.WIN_CLOSED or event == 'Cancel':  # if user closes window or clicks cancel
                break
        window.close()
