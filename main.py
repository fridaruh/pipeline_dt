from utils import Utils
from models import Models

if __name__ == "__main__":

    utils = Utils() #Para inicializar una clase solo hay que llamar a su constructor
    models = Models()

    data = utils.load_from_csv('./in/income.csv')

    print('Aquí tienes una muestra de tus datos:')
    print(' ')

    print(data.head(5))

######## Creación de gráficas #############

# grafica_barras(dataset, columna)

    utils.grafica_barras(data, 'workclass')


########### Creación de modelo y evaluación ##############

## features_target(dataset, drop_cols, cols_wanted, y)
    dropear = ['Unnamed: 0','fnlwgt','capital-gain', 'capital-loss','income','native-country']
    dummies = ['workclass','education','marital-status','occupation','relationship','race','sex']
    target = ['income_bi']



    X, y = utils.features_target(data, dropear, dummies, target)

    print('Estamos entrenando el modelo...')

    resultados = models.tree_training(X,y)
    print('')
    print(f'El score del modelo en train es: {resultados[0]}')
    print(f'El score del modelo en test es: {resultados[1]}')

