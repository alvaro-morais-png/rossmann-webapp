import pickle
import inflection
import pandas as pd
import numpy as np
import math
import datetime

class Rossmann( object ):
    def __init__( self ):
       self.home_path = ''
       self.competition_distance_scaler   = pickle.load( open( self.home_path + 'parameter/competition_distance_scaler.pkl', 'rb'))
       self.competition_time_month_scaler = pickle.load( open( self.home_path + 'parameter/competition_time_month_scaler.pkl', 'rb'))
       self.promo_time_week_scaler        = pickle.load( open( self.home_path + 'parameter/promo_time_week_scaler.pkl', 'rb'))
       self.year_scaler                   = pickle.load( open( self.home_path + 'parameter/year_scaler.pkl', 'rb'))
       self.store_type_scaler             = pickle.load( open( self.home_path + 'parameter/store_type_scaler.pkl', 'rb'))
       self.assortment_scaler             = pickle.load( open( self.home_path + 'parameter/assortment_scaler.pkl', 'rb'))

    def data_cleaning( self, df1 ):
        ## 1.1 Rename columns
        cols_old = [['Store', 'DayOfWeek', 'Date', 'Open', 'Promo',
                    'StateHoliday', 'SchoolHoliday', 'StoreType', 'Assortment',
                    'CompetitionDistance', 'CompetitionOpenSinceMonth',
                    'CompetitionOpenSinceYear', 'Promo2', 'Promo2SinceWeek',
                    'Promo2SinceYear', 'PromoInterval']]

        snakecase = lambda x: inflection.underscore( x )
        cols_new = list(map(snakecase, cols_old[0]))

        #rename
        df1.columns = cols_new

        ## 1.3 Data Types
        #transformou 'date' em datetime e verificou os tipos das colunas
        df1['date']= pd.to_datetime(df1['date'])

        ## 1.5 Fillout NA
        # competition_distance
        #pegar a coluna competition_distance
        #verificar se tem valor nulo
        #se tiver valor nulo, substituir por 200000.0 (para simular que a loja não tem concorrência próxima)
        #se não tiver valor nulo, manter o valor da coluna
        df1['competition_distance'] = df1['competition_distance'].apply(lambda x: 200000.0 if math.isnan(x) else x)

        # competition_open_since_month - substituir os valores nulos pelo mês da data da linha para não ter valores nulos
        #pegar a coluna competition_open_since_month
        #verificar se tem valor nulo
        #se tiver valor nulo, substituir pelo valor do mês da data da linha 
        #se não tiver valor nulo, manter o valor da coluna
        df1['competition_open_since_month'] = df1.apply(lambda x: x['date'].month if math.isnan(x['competition_open_since_month']) else x['competition_open_since_month'], axis=1)

        # competition_open_since_year
        df1['competition_open_since_year'] = df1.apply(lambda x: x['date'].year if math.isnan(x['competition_open_since_year']) else x['competition_open_since_year'], axis=1)

        # promo2_since_week
        df1['promo2_since_week'] = df1.apply(lambda x: x['date'].week if math.isnan(x['promo2_since_week']) else x['promo2_since_week'], axis=1)

        #promo2_since_year
        df1['promo2_since_year'] = df1.apply(lambda x: x['date'].year if math.isnan(x['promo2_since_year']) else x['promo2_since_year'], axis=1)

        # promo_interval
        #transformou os valores nulos da coluna promo_interval em zero, para não ter valor nulo
        #se a coluna promo_interval for igual a 0, então is_promo é igual a 0
        #se a coluna month_map estiver presente na coluna promo_interval, então is_promo é igual a 1
        #se não, is_promo é igual a 0
        #criou uma coluna nova chamda is_promo, que indica se a loja está em promoção ou não, com base na coluna promo_interval e no mês da data da linha
        month_map ={ 1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec' }
        df1['promo_interval']=df1['promo_interval'].fillna(0)#atualizei o código, o .fillna(0, inplace=True) não funcionou bem então usei o .fillna(0) e atribui o resultado para a própria coluna promo_interval
        df1['month_map'] = df1['date'].dt.month.map(month_map)
        df1['is_promo']= df1[['promo_interval', 'month_map']].apply(lambda x: 0 if x['promo_interval'] == 0 else 1 if x['month_map'] in x['promo_interval'].split(',') else 0, axis=1)
        
        ## 1.6 Change Types
        # Após preencher os valores nulos, é necessário converter as colunas necessárias para o tipo inteiro, 
        # para evitar problemas futuros na análise e modelagem dos dados. As colunas que precisam ser convertidas são:
        # competition_open_since_month, competition_open_since_year, promo2_since_week e promo2_since_year.
        df1['competition_open_since_month'] = df1['competition_open_since_month'].astype(int)
        df1['competition_open_since_year'] = df1['competition_open_since_year'].astype(int)
        df1['promo2_since_week'] = df1['promo2_since_week'].astype(int)
        df1['promo2_since_year'] = df1['promo2_since_year'].astype(int)

        return df1

    def feature_engineering( self, df2 ):
        # ── Extração de componentes da data ─────────────────────────
        # Extrai ano, mês, dia e semana a partir da coluna date
        # para permitir análises temporais granulares

        df2['year']  = df2['date'].dt.year   # ano
        df2['month'] = df2['date'].dt.month  # mês (1-12)
        df2['day']   = df2['date'].dt.day    # dia do mês

        # Extraçãoo da semana do ano
        # dt.weekofyear foi descontinuada no pandas 2.x
        # substituta: isocalendar().week (padrão ISO 8601)
        df2['week_of_year'] = df2['date'].dt.isocalendar().week

        # Combinação de ano e semana no formato 'YYYY-WW'
        # útil para agrupamentos e análises semanais
        df2['year_week'] = df2['date'].dt.strftime('%Y-%W')

        # competition since
        # ── Tempo desde abertura do competidor ──────────────────────
        # Une as colunas competition_open_since_year e competition_open_since_month
        # em uma única data (dia fixado em 1) usando datetime.datetime
        df2['competition_since'] = df2.apply(lambda x: datetime.datetime(year=x['competition_open_since_year'], month = x['competition_open_since_month'], day=1 ), axis=1)

        # Calcula quantos meses se passaram desde a abertura do competidor
        # divide a diferença de datas por 30 para converter em meses
        df2['competition_time_month'] = ((df2['date'] - df2['competition_since'])/30).apply(lambda x: x.days).astype(int)

        # promo since
        # ── Tempo desde início da promoção extendida (promo2) ───────
        # Une promo2_since_year e promo2_since_week no formato 'YYYY-WW'
        # para reconstruir a data de início da promoção
        df2['promo_since'] = df2['promo2_since_year'].astype(str) + '-' + df2['promo2_since_week'].astype(str)

        # Converte a string 'YYYY-WW' para datetime
        # o sufixo '-1' ancora a data na segunda-feira da semana
        # subtrai 7 dias para alinhar ao início real da semana da promoção
        df2['promo_since'] = df2['promo_since'].apply(lambda x: datetime.datetime.strptime(x + '-1', '%Y-%W-%w') - datetime.timedelta(days=7))

        # Calcula quantas semanas se passaram desde o início da promoção
        # valores positivos = promoção ativa | valores negativos = antes da promoção
        df2['promo_time_week'] = ((df2['date'] - df2['promo_since'])/7).apply(lambda x: x.days).astype(int)

        # ── Decodificação de variáveis categóricas ──────────────────
        # Traduz os códigos originais para descrições legíveis
        # facilitando interpretação e análise exploratória

        # assortment: a=basic | b=extra | c=extended
        # assortment
        df2['assortment'] = df2['assortment'].apply(lambda x: 'basic' if x == 'a' else 'extra' if x == 'b' else 'extended')


        # state_holiday: a=feriado público | b=páscoa | c=natal | 0=dia normal
        df2['state_holiday']=df2['state_holiday'].apply(lambda x: 'public_holiday' if x == 'a' else 'easter_holiday' if x == 'b' else 'christmas' if x == 'c' else 'regular_day')
        
        # 3.0 FILTRAGEM DE VARIÁVEIS
        ## 3.1 Filtragem das linhas
        # restrinções de negócio
        df2 = df2[df2['open'] != 0]
        ## 3.2 Seleção das Colunas
        cols_drop = ['open', 'month_map']#'promo' foi recolocada no dataset
        df2 = df2.drop(cols_drop, axis=1)

        return df2
    
    def data_preparation( self, df5):
            
        # analisando o item 4.1.2 Numerical Variable identificou-se que as variáveis 
        # numéricas não se enquandrão em distribuição normal, e optou-se a utilizar o Rescaling
        ## 5.2 Rescaling
        

        # competition_distance
        df5['competition_distance'] = self.competition_distance_scaler.fit_transform(df5[['competition_distance']].values )


        # competition_time_month
        df5['competition_time_month'] = self.competition_time_month_scaler.fit_transform(df5[['competition_time_month']].values )
        
        # promo_time_week
        df5['promo_time_week'] = self.promo_time_week_scaler.fit_transform(df5[['promo_time_week']].values )

        # year
        df5['year'] = self.year_scaler.fit_transform(df5[['year']].values )

        ## 5.3 Transformacao
        ### 5.3.1 Encoding
        # state_holiday - ONE HOT ENCODING
        df5 = pd.get_dummies(df5, prefix=['state_holiday'], columns = ['state_holiday'], dtype = int )

        # store_type - LABEL ENCODER
        df5['store_type'] = self.store_type_scaler.fit_transform(df5['store_type'])
        
        # assortment - ORDINAL ENCONDING # dá pra fazer o projeto ppor dois meios com sklearn ou map(dict)
        df5['assortment'] = self.assortment_scaler.fit_transform(df5[['assortment']])

        ### 5.3.3. Nature Transformation
        # transformando variáveis de natureza cíclica
        # Day
        df5['day_sin'] = df5[ 'day' ].apply( lambda x: np.sin( x * ( 2. * np.pi/30 ) ) )
        df5['day_cos'] = df5[ 'day' ].apply( lambda x: np.cos( x * ( 2. * np.pi/30 ) ) )

        # day_of_week
        df5['day_of_week_sin'] = df5[ 'day_of_week' ].apply( lambda x: np.sin( x * ( 2. * np.pi/7 ) ) )
        df5['day_of_week_cos'] = df5[ 'day_of_week' ].apply( lambda x: np.cos( x * ( 2. * np.pi/7 ) ) )

        # month
        df5['month_sin'] = df5[ 'month' ].apply( lambda x: np.sin( x * ( 2. * np.pi/12 ) ) )
        df5['month_cos'] = df5[ 'month' ].apply( lambda x: np.cos( x * ( 2. * np.pi/12 ) ) )

        # week_of_year
        df5['week_of_year_sin'] = df5[ 'week_of_year' ].apply( lambda x: np.sin( x * ( 2. * np.pi/52 ) ) )
        df5['week_of_year_cos'] = df5[ 'week_of_year' ].apply( lambda x: np.cos( x * ( 2. * np.pi/52 ) ) )

        cols_selected = ['store', 'promo', 'store_type', 'assortment', 'competition_distance', 
                 'competition_open_since_month', 'competition_open_since_year', 'promo2', 
                 'promo2_since_week', 'promo2_since_year', 'competition_time_month',
                 'promo_time_week', 'day_sin', 'day_cos', 'day_of_week_sin', 
                 'day_of_week_cos', 'month_sin', 'month_cos', 'week_of_year_sin', 
                 'week_of_year_cos']

        return df5[ cols_selected ]
    
    def get_prediction( self, model, original_data, test_data ):
        # prediction
        pred = model.predict(test_data)

        # join pred into the original data
        original_data['prediction'] = np.expm1(pred)

        return original_data.to_json(orient = 'records', date_format = 'iso' )