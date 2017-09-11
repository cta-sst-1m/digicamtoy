

def gain_drop(nsb_rate, cell_capacitance=85. * 1E-15, bias_resistance = 10. * 1E3):

    return 1. / (1. + nsb_rate * cell_capacitance * bias_resistance * 1E9)
