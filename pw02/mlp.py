import numpy as np
import matplotlib.pyplot as pl
from ipywidgets import interact, widgets
from tempfile import NamedTemporaryFile
from IPython.display import HTML, display
import base64

#---------------------------------------------------------------------------------
VIDEO_TAG = """<video controls>
 <source src="data:video/x-m4v;base64,{0}" type="video/mp4">
 Your browser does not support the video tag.
</video>"""

def anim_to_html(anim):
    if not hasattr(anim, '_encoded_video'):
        with NamedTemporaryFile(suffix='.mp4') as f:
            anim.save(f.name, fps=20, extra_args=['-vcodec', 'libx264'])
            video = open(f.name, "rb").read()
        # video.encode generates error: 'bytes' object has no attribute 'encode'
        #anim._encoded_video = video.encode("base64")
        
        # workaround: encode video with base64 lib
        anim._encoded_video = base64.b64encode(video)
    
    # base64 object has format b'......'
    # Decode it with utf-8 to remove the b'' and keep only the encoded part
    return VIDEO_TAG.format(anim._encoded_video.decode('utf-8'))

def display_animation(anim):
    pl.close(anim._fig)
    display(HTML(anim_to_html(anim)))

#- Activation functions ----------------------------------------------------------
def linear(neta):
    '''Linear activation function'''
    output = neta
    d_output = np.ones(len(neta))
    return (output, d_output)

def sigmoid(neta):
    '''Sigmoidal activation function'''
    output = 1 / (1 + np.exp(-neta))
    d_output = output * (1 - output)
    return (output, d_output)

def htan(neta):
    '''Hyperbolic tangent activation function'''
    exp = np.exp(neta)
    m_exp = np.exp(-neta)
    output = (exp - m_exp ) / (exp + m_exp)
    d_output = 1 - (output * output)
    return (output, d_output)

def perceptron(input_values, weights, bias, activation_function):
    '''Computes the output of a perceptron
    :param input_values: inputs to the perceptron
    :param weights: perceptron parameters (multiply inputs)
    :param bias: perceptron parameter (adds to inputs)
    :param activation_function: activation function to apply to the weighted sum of inputs
    :return: perceptron output'''
    neta = np.dot(input_values, weights) + bias
    output, d_output = activation_function(neta)
    return output

def feedforward(input_values, weights, bias, activation_function):
    w_x_0 = weights[0]
    w_x_1 = weights[1]
    w_y_0 = weights[2]
    w_y_1 = weights[3]
    w_h_0 = weights[4]
    w_h_1 = weights[5]
    b_0   = bias[0]
    b_1   = bias[1]
    b_h   = bias[2]
    h_0 = perceptron(input_values, [w_x_0, w_y_0], b_0, activation_function)
    h_1 = perceptron(input_values, [w_x_1, w_y_1], b_1, activation_function)
    h = np.array([h_0, h_1]).T
    return perceptron(h, [w_h_0, w_h_1], b_h, activation_function)
    
#- Plotting ---------------------------------------------------------------------------
class MLPPlotter2D:
    def create_slider(slef, name):
        return widgets.FloatSlider(
            value=0.5,
            min=-2.0,
            max=2.0,
            step=0.01,
            description=name,
        )

    def create_controls(self):
        controls = {name:self.create_slider(name) for name in ['w_x_0', 'w_x_1', 'w_y_0', 'w_y_1', 'w_h_0', 'w_h_1', 'b_0', 'b_1', 'b_h']}

        controls['activation_function_index'] = widgets.Dropdown(
            options={k:i for i,k in enumerate(self.activation_functions_dict.keys())},
            #options={self.activation_functions_dict.keys()[i]:i for i in range(len(self.activation_functions_dict))},
            value=1,
            description='Activation function:',
        )
        return controls

    def display_controls(self):
        p0 = widgets.HBox(children=[self.controls['w_x_0'], self.controls['w_y_0'], self.controls['b_0']])
        p1 = widgets.HBox(children=[self.controls['w_x_1'], self.controls['w_y_1'], self.controls['b_1']])
        h0 = widgets.HBox(children=[self.controls['w_h_0'], self.controls['w_h_1'], self.controls['b_h']])

        #widgets.interactive(plot_MLP, **controls);
        display(p0)
        display(p1)
        display(h0)
        display(self.controls['activation_function_index'])
        #plot_MLP(**{key:controls[key].value for key in controls.keys()})

    def __init__(self, xlim=(-1.2,1.2), ylim=(-1.2,1.2), data=None):
        self.xlim=xlim
        self.ylim=ylim
        input_x = np.arange(xlim[0], xlim[1], 0.1)
        input_y = np.arange(ylim[0], ylim[1], 0.1)
        self.input_x_matrix, self.input_y_matrix = np.meshgrid(input_x, input_y)
        self.inputs_xy = np.concatenate((self.input_x_matrix.flatten()[:,np.newaxis], self.input_y_matrix.flatten()[:,np.newaxis]), axis=1)
        self.activation_functions_dict = {'Linear': linear, 'Sigmoid': sigmoid, 'Hyperbolic tangent': htan}
        self.data = data
        if len(data) > 0:
            self.c1_i = data[:,2] > 0
            self.c2_i = data[:,2] < 0
        self.error = []
        self.ax_line = None
        self.ax_im = None

        self.controls = self.create_controls()
        
    def plot_once(self):
        self.plot_interactive(**{key:self.controls[key].value for key in self.controls.keys()})
        
    def plot_interactive(self, w_x_0, w_x_1, w_y_0, w_y_1, w_h_0, w_h_1, b_0, b_1, b_h, activation_function_index):
        w_0 = np.array([w_x_0, w_y_0])
        w_1 = np.array([w_x_1, w_y_1])
        w_h = np.array([w_h_0, w_h_1])
       
        #activation_function = self.activation_functions_dict.get(self.activation_functions_dict.keys()[activation_function_index])
        activation_function = self.activation_functions_dict.get(list(self.activation_functions_dict.keys())[activation_function_index])
        h_0 = perceptron(self.inputs_xy, w_0, b_0, activation_function)
        h_1 = perceptron(self.inputs_xy, w_1, b_1, activation_function)
        
        output_values = perceptron(np.vstack((h_0, h_1)).T, w_h, b_h, activation_function)
        output_matrix = np.reshape(output_values, self.input_x_matrix.shape)
        
        pl.figure(figsize=(12,6))
        pl.subplot(121)
        pl.imshow(np.flipud(output_matrix), interpolation='None', extent=(self.xlim[0], self.xlim[1], self.ylim[0], self.ylim[1]), vmin=-1, vmax=1)
        pl.colorbar(shrink=0.7)
        if len(self.data) > 0:
            pl.scatter(x=self.data[self.c1_i,0], y=self.data[self.c1_i,1], c='r', s=100, lw=0)
            pl.scatter(x=self.data[self.c2_i,0], y=self.data[self.c2_i,1], c='b', s=100, lw=0)
        pl.xlabel('x')
        pl.ylabel('y')
        pl.grid()
        pl.title('Perceptron output')
        if len(self.data) > 0:
            inputs = self.data[:,0:2]
            targets = self.data[:,2]
            h_0 = perceptron(inputs, w_0, b_0, activation_function)
            h_1 = perceptron(inputs, w_1, b_1, activation_function)
            output_values = perceptron(np.vstack((h_0, h_1)).T, w_h, b_h, activation_function)
            self.error.append(np.mean(np.power((output_values - targets), 2)))
            pl.subplot(122)
            pl.plot(self.error)
            pl.xlabel('Iterations')
            pl.ylabel('MSE')
            pl.grid()
            pl.title('Perceptron error')

    def init_animation(self):
        if not self.ax_im:
            self.ax_im = pl.subplot(121)
            pl.xlabel('x')
            pl.ylabel('y')
            pl.title('Perceptron output')
            pl.grid()
        self.im = self.ax_im.imshow(np.zeros(self.input_x_matrix.shape), interpolation='None', vmin=-1, vmax=1, extent=(self.xlim[0], self.xlim[1], self.ylim[0], self.ylim[1]))
        pl.colorbar(self.im, shrink=0.7)
        self.ax_im.scatter(x=self.data[self.c1_i,0], y=self.data[self.c1_i,1], c='r', s=100, lw=0)
        self.ax_im.scatter(x=self.data[self.c2_i,0], y=self.data[self.c2_i,1], c='b', s=100, lw=0)

        if not self.ax_line:
            self.ax_line = pl.subplot(122)
            pl.xlabel('Iterations')
            pl.ylabel('MSE')
            pl.title('Perceptron error')
            pl.grid()
        self.line, = self.ax_line.plot([],[])
        self.line.set_data([], [])
    
    def data2animation(self, i, inputs, weights, bias, targets, activation_function):
        output_values = feedforward(self.inputs_xy, weights, bias, activation_function)
        output_matrix = np.reshape(output_values, self.input_x_matrix.shape)
        self.im.set_data(np.flipud(output_matrix))

        output_values = feedforward(inputs, weights, bias, activation_function)
        self.error.append(np.mean((output_values - targets) ** 2))
        x = np.arange(len(self.error))
        self.line.set_data(x, np.array(self.error))
        self.ax_line.set_xlim((0, max(1, i)))
        self.ax_line.set_ylim((0, max(self.error)))
        return self.line, self.im

