import numpy as np
import pylab
import re

# cities in our weather data
CITIES = [
    'BOSTON',
    'SEATTLE',
    'SAN DIEGO',
    'PHILADELPHIA',
    'PHOENIX',
    'LAS VEGAS',
    'CHARLOTTE',
    'DALLAS',
    'BALTIMORE',
    'SAN JUAN',
    'LOS ANGELES',
    'MIAMI',
    'NEW ORLEANS',
    'ALBUQUERQUE',
    'PORTLAND',
    'SAN FRANCISCO',
    'TAMPA',
    'NEW YORK',
    'DETROIT',
    'ST LOUIS',
    'CHICAGO'
]

INTERVAL_1 = list(range(1961, 2006))
INTERVAL_2 = list(range(2006, 2016))


class Climate(object):
    """
    The collection of temperature records loaded from given csv file
    """

    def __init__(self, filename):
        """
        Initializes a Climate instance, which stores the temperature records
        loaded from a given csv file specified by filename.

        Args:
            filename: name of the csv file (str)
        """
        self.rawdata = {}

        f = open(filename, 'r')
        header = f.readline().strip().split(',')
        for line in f:
            items = line.strip().split(',')

            date = re.match('(\d\d\d\d)(\d\d)(\d\d)',
                            items[header.index('DATE')])
            year = int(date.group(1))
            month = int(date.group(2))
            day = int(date.group(3))

            city = items[header.index('CITY')]
            temperature = float(items[header.index('TEMP')])
            if city not in self.rawdata:
                self.rawdata[city] = {}
            if year not in self.rawdata[city]:
                self.rawdata[city][year] = {}
            if month not in self.rawdata[city][year]:
                self.rawdata[city][year][month] = {}
            self.rawdata[city][year][month][day] = temperature

        f.close()

    def get_yearly_temp(self, city, year):
        """
        Get the daily temperatures for the given year and city.

        Args:
            city: city name (str)
            year: the year to get the data for (int)

        Returns:
            a numpy 1-d array of daily temperatures for the specified year and
            city
        """
        temperatures = []
        assert city in self.rawdata, "provided city is not available"
        assert year in self.rawdata[city], "provided year is not available"
        for month in range(1, 13):
            for day in range(1, 32):
                if day in self.rawdata[city][year][month]:
                    temperatures.append(self.rawdata[city][year][month][day])
        return np.array(temperatures)

    def get_daily_temp(self, city, month, day, year):
        """
        Get the daily temperature for the given city and time (year + date).

        Args:
            city: city name (str)
            month: the month to get the data for (int, where January = 1,
                December = 12)
            day: the day to get the data for (int, where 1st day of month = 1)
            year: the year to get the data for (int)

        Returns:
            A float of the daily temperature for the specified time (year +
            date) and city
        """
        assert city in self.rawdata, "provided city is not available"
        assert year in self.rawdata[city], "provided year is not available"
        assert month in self.rawdata[city][year], "provided month is not available"
        assert day in self.rawdata[city][year][month], "provided day is not available"
        return self.rawdata[city][year][month][day]


def generate_models(x, y, degs):
    """
    Generates regression models by fitting a polynomial for each degree in degs
    to points (x, y).
    Args:
        x: a list with length N, representing the x-coords of N sample points
        y: a list with length N, representing the y-coords of N sample points
        degs: a list of degrees of the fitting polynomial
    Returns:
        a list of numpy arrays, where each array is a 1-d array of coefficients
        that minimizes the squared error of the fitting polynomial
    """
    xVals = pylab.array(x)
    yVals = pylab.array(y)
    models = []

    for ye_like_degs in degs:
        model = pylab.polyfit(xVals, yVals, ye_like_degs)
        models.append(model)

    return models


def r_squared(y, estimated):
    """
    Calculates the R-squared error term.
    Args:
        y: list with length N, representing the y-coords of N sample points
        estimated: a list of values estimated by the regression model
    Returns:
        a float for the R-squared error term
    """
    sum_of_squares = 0
    for i in range(len(estimated)):
        sum_of_squares += (estimated[i] - y[i])**2
    variance = sum_of_squares / len(estimated)
    return 1 - (variance / np.var(y))


def evaluate_models_on_training(x, y, models):
    """
    For each regression model, computes the R-square for this model with the
    standard error over slope of a linear regression line (only if the model is
    linear), and plot the data along with the best fit curve.

    Plots data points (x,y) as blue dots and the best
    fit curve (aka model) as a red solid line. 

    Labels:
        Degree of regression model.
        R-square of model evaluated on the given data points.

    Args:
        x: a list of length N, representing the x-coords of N sample points
        y: a list of length N, representing the y-coords of N sample points
        models: A list containing the regression models to apply to data. Each model is a numpy array storing the coefficients of
            a polynomial.

    Returns:
        None
    """
    for model in models:
        # estimates == estYVals
        estimates = pylab.polyval(model, x)

        if len(model) == 2:
            r_sqr = r_squared(y, estimates)

        pylab.plot(x, y, 'bo', label='recorded y values')
        pylab.plot(x, estimates, 'r', label='Regression line')
        pylab.legend(loc='best')
        pylab.title('climate changes\n' + 'r squared value:' + str(r_sqr))
        pylab.xlabel('years')
        pylab.ylabel('temperature')


# Beginning of program
raw_data = Climate('data.csv')

# Charts daily temp on a given day, over INTERVAL yrs, for a city.
y1 = []
for year in INTERVAL_1:
    y1.append(raw_data.get_daily_temp('BOSTON', 1, 10, year))
models = generate_models(INTERVAL_1, y1, [1])
evaluate_models_on_training(INTERVAL_1, y1, models)

# Charts average yearly temp, over INTERVAL yrs, for a city.
y2 = []
for year in INTERVAL_1:
    temp_array = raw_data.get_yearly_temp('BOSTON', year)
    avg_temp = temp_array.sum() / len(temp_array)
    y2.append(avg_temp)
models = generate_models(INTERVAL_1, y2, [1])
evaluate_models_on_training(INTERVAL_1, y2, models)
