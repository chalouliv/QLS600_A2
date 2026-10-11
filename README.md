# QLS600_A2
QLS600 A2


Run 1 results

data: CookLab1UnknownChannelExtra.txt   n_train=1921  n_valid=339   restarts=5  workers=18  trials=1
estimated measurement-noise SD (flat-V windows): 0.00489
  -> no honest model can score below this on new data.

trial 1/1: dispatching 75 fits...
  trial 1 best: m^2 h^2 n^2 valid=0.00683

all trials done.

structure                      valid_mean valid_std valid_min  R2_mean  BIC_mean
m^2 h^2 n^2                       0.00683   0.00000   0.00683   0.9087  -19164.0
m^1 h^2 n^1                       0.00711   0.00000   0.00711   0.9012  -19051.4
m^3 h^2 n^1                       0.00730   0.00000   0.00730   0.8959  -19482.0
m^2 h^2 n^1                       0.00732   0.00000   0.00732   0.8954  -19481.8
m^3 h^1 n^1                       0.00745   0.00000   0.00745   0.8916  -19495.5
m^2 h^1 n^1                       0.00748   0.00000   0.00748   0.8907  -19497.6
m^1 h^1 n^1                       0.00782   0.00000   0.00782   0.8804  -19516.7
m^2 h^1 n^2                       0.00783   0.00000   0.00783   0.8802  -19433.1
m^3 h^2 n^2                       0.00799   0.00000   0.00799   0.8752  -19030.6
m^1 h^1 n^2                       0.00807   0.00000   0.00807   0.8728  -19460.9
m^3 h^1 n^2                       0.00825   0.00000   0.00825   0.8669  -19442.0
m^2 h^1                           0.01043   0.00000   0.01043   0.7875  -19574.2
m^1 h^1                           0.01046   0.00000   0.01046   0.7864  -19586.8
m^3 h^1                           0.01046   0.00000   0.01046   0.7860  -19565.8
m^1 h^2 n^2                       0.01273   0.00000   0.01273   0.6832  -18278.4

WINNER by held-out RMSE (mean): m^2 h^2 n^2  valid=0.00683 +/- 0.00000
WINNER by BIC (mean):           m^1 h^1  BIC=-19586.8
held-out RMSE is 1.40x the noise floor: structural gap remains (or noisier regime).

--- exponent discriminator: fit p as a free parameter ---
  start 0: p_hat = 0.500
  start 1: p_hat = 0.500
  start 2: p_hat = 0.500
  start 3: p_hat = 0.500
  start 4: p_hat = 0.500
  median p_hat = 0.500  ->  inconclusive
