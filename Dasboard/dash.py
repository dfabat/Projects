# pacotes de manipulação e visualização simples
import pandas as pd
import streamlit as st
from prophet import Prophet
import cufflinks as cf
import plotly.express as px
import plotly.graph_objects as go
from prophet.plot import plot_plotly
####
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
cf.go_offline()
###################################################################################

# Título
st.title("Estudo da variação da temperatura planetária")
st.write("Criado por Diego Batista, Rogério Chinen, Tsuyoshi Fukuda")

# Avisos importantes
st.markdown("**Avisos importantes:**")
## Aviso 1
st.write(f"As análises abaixo foram geradas a partir do conjunto de dados **'Climate Change: Earth Surface Temperature Data'**. Para mais informações, acesse:")
link_kaggle = '[Climate Change: Earth Surface Temperature Data](https://www.kaggle.com/berkeleyearth/climate-change-earth-surface-temperature-data)'
st.markdown(link_kaggle, unsafe_allow_html=True)
## Aviso 2
st.write("Informações sobre continentes e códigos dos países foram obtidas no DataHub.io, postado por **JohnSnowLabs**. Para mais informações, acesse:")
link_datahub = "[Country and Continent Codes List](https://datahub.io/JohnSnowLabs/country-and-continent-codes-list#data)"
st.markdown(link_datahub, unsafe_allow_html=True)
##############################################################################################################

# Temperatura média na terra e oceano
st.subheader("Variações na temperatura global ao longo dos anos")

# local dos arquivos

## dataframes
def read_dataframe(name, bool=False):
    df = pd.read_csv(name)
    if bool == True:
        df.dt = pd.to_datetime(df.dt)
        return df
    else:
        return df


df_land = read_dataframe("https://raw.githubusercontent.com/dfabat/Projects/main/Dasboard/df_land.csv", True)
df_ocean = read_dataframe("https://raw.githubusercontent.com/dfabat/Projects/main/Dasboard/df_ocean.csv", True)

## Plotagem dos gráficos
def plot_land(df_land):
    
    # formatação do hover
    text_1 = [f'Ano: {x}<br>Temperatura: {str(round(y,1)).replace(".", ",")}°C'
     for x, y in zip(df_land.dt.dt.year.tolist(),
                    df_land.LandAverageTemperature.tolist())]
    
    text_2 = [f'Ano: {x}<br>Temperatura: {str(round(y,1)).replace(".", ",")}°C'
     for x, y in zip(df_land.dt.dt.year.tolist(),
                    df_land.LandMovingAverageTemperature12.tolist())]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_land["dt"], y=round(df_land["LandAverageTemperature"],1), name='Temperatura média mensal', mode="markers", marker=dict(size = 3.5), opacity = 0.5, text=text_1, hoverinfo="text"))

    fig.add_trace(go.Scatter(
        x=df_land['dt'], y=round(df_land["LandMovingAverageTemperature12"],1), name='Temperatura média anual móvel', text=text_2, hoverinfo="text"))

    fig.layout.update(title_text='Temperatura global média dos continentes',
                     xaxis_rangeslider_visible=True
                     )

    fig.layout.update(
        xaxis=dict(
                showline=True,
                showgrid=False,
                showticklabels=True,
                linecolor='rgb(0, 0, 0)',
                linewidth=2,
                ticks='outside',
                title_text = "Ano",
                tickfont=dict(
                    family='Arial',
                    size=12,
                    color='rgb(82,82,82)'),
                title_standoff = 200,
            ),
        yaxis=dict(
                showline=True,
                showgrid=False,
                zeroline=True,
                showticklabels=True,
                ticks='outside',
                title_text = "Temperatura em °C",
                linecolor='rgb(0, 0, 0)',
                linewidth=2,
        ),
        legend=dict(
            orientation= "h",
            title=dict(
                font=dict(
                    color="#000000",
                    family="arial")),
            x = 0.08, # move a legenda horizontalmente
            y=1.05 # move a legenda verticalmente
    ),
        margin=dict( # altera as margens do gráfico
        autoexpand=False,
        r=20,
        l=100,
        t=100,
        b=110
    ),
        plot_bgcolor='rgb(255,255,255)',  # plot_bgcolor='white'   
        
        # alteração das cores das linhas do gráfico
        colorway=["#ff7f0e", "#162dc4"], # cores dos gráficos
    )
    st.plotly_chart(fig, use_container_width=True)
    
def plot_ocean(df_ocean):
    # formatação do hover
    text_1 = [f'Ano: {x}<br>Temperatura: {str(round(y,1)).replace(".", ",")}°C'
     for x, y in zip(df_ocean.dt.dt.year.tolist(),
                    df_ocean.LandAndOceanAverageTemperature.tolist())]
    
    text_2 = [f'Ano: {x}<br>Temperatura: {str(round(y,1)).replace(".", ",")}°C'
     for x, y in zip(df_ocean.dt.dt.year.tolist(),
                    df_ocean.LandAndOceanMovingAverageTemperature12.tolist())]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_ocean['dt'], y=round(df_ocean['LandAndOceanAverageTemperature'],1), name='Temperatura média mensal', mode="markers", marker=dict(size = 3.5), opacity = 0.5, text=text_1, hoverinfo="text"))

    fig.add_trace(go.Scatter(
        x=df_ocean['dt'], y=round(df_ocean['LandAndOceanMovingAverageTemperature12'],1), name='Temperatura média anual móvel', text=text_2, hoverinfo="text"))

    fig.layout.update(title_text='Temperatura global média dos continentes e oceanos',
                     xaxis_rangeslider_visible=True)

    fig.layout.update(
        xaxis=dict(
                showline=True,
                showgrid=False,
                showticklabels=True,
                linecolor='rgb(0, 0, 0)',
                linewidth=2,
                ticks='outside',
                title_text = "Ano",
                tickfont=dict(
                    family='Arial',
                    size=12,
                    color='rgb(82,82,82)'),
                title_standoff = 200,
            ),
        yaxis=dict(
                showline=True,
                showgrid=False,
                zeroline=True,
                showticklabels=True,
                ticks='outside',
                title_text = "Temperatura em °C",
                linecolor='rgb(0, 0, 0)',
                linewidth=2
            ),
        legend=dict(
            orientation= "h",
            title=dict(
                font=dict(
                    color="#000000",
                    family="arial")),
            x = 0.08, # move a legenda horizontalmente
            y=1.05 # move a legenda verticalmente
    ),
        margin=dict( # altera as margens do gráfico
        autoexpand=False,
        r=20,
        l=100,
        t=100,
        b=110
    ),
        plot_bgcolor='rgb(255,255,255)',  # plot_bgcolor='white'   
        
        # alteração das cores das linhas do gráfico
        colorway=["#162dc4", "#cf2c17"], # cores dos gráficos
    )
    st.plotly_chart(fig, use_container_width=True)
    
# checknox
is_land = st.radio("Escolha abaixo o gráfico das temperaturas médias globais para avaliação", ("Continentes", "Continentes + Oceanos"))
#st.write(is_land)

if is_land == "Continentes":
    plot_land(df_land)
elif is_land == "Continentes + Oceanos":
    plot_ocean(df_ocean)
else:
    pass

##############################################################################################################

# temperatura continentes
st.subheader("Temperaturas médias nos continentes")

# Temperatura das cidades de 1901 a 2013
df_city = read_dataframe("https://raw.githubusercontent.com/dfabat/Projects/main/Dasboard/Dataset_limpo_reduzido.csv")

def continentes(df_continent):
    # Gerando os modelos para o hover
    text = [f'Continente: {x}<br>Ano: {y}<br>Temperatura: {str(round(z,2)).replace(".", ",")}°C'
            for x, y, z in zip(df_continent.Continent_Name.tolist(), df_continent.Year.tolist(),
                               df_continent.AverageTemperature.tolist())]

    # Gerando a visualização gráfica
    fig = go.Figure()

    # criando o plot para a África
    fig.add_trace(go.Scatter(x=df_continent.Year[df_continent.Continent_Name == "Africa"],
                             y=df_continent.AverageTemperature[df_continent.Continent_Name == "Africa"],
                             mode="lines+markers", name="Africa", text=text[:12], hoverinfo='text'))

    # Criando o plot para a Ásia
    fig.add_trace(go.Scatter(x=df_continent.Year[df_continent.Continent_Name == "Asia"],
                             y=df_continent.AverageTemperature[df_continent.Continent_Name == "Asia"],
                             mode="lines+markers", name="Asia", text=text[12:24], hoverinfo='text'))

    # Criando o plot para a Europa
    fig.add_trace(go.Scatter(x=df_continent.Year[df_continent.Continent_Name == "Europe"],
                             y=df_continent.AverageTemperature[df_continent.Continent_Name == "Europe"],
                             mode="lines+markers", name="Europa", text=text[24:36], hoverinfo='text'))

    # Criando o plot para a Oceania
    fig.add_trace(go.Scatter(x=df_continent.Year[df_continent.Continent_Name == "Oceania"],
                             y=df_continent.AverageTemperature[df_continent.Continent_Name == "Oceania"],
                             mode="lines+markers", name="Oceania", text=text[48:60], hoverinfo='text'))

    # Criando o plot para a América do Norte e Central
    fig.add_trace(go.Scatter(x=df_continent.Year[df_continent.Continent_Name == "North America"],
                             y=df_continent.AverageTemperature[df_continent.Continent_Name == "North America"],
                             mode="lines+markers", name="América do Norte", text=text[36:48], hoverinfo='text'))

    # Criando o plot para a América do Sul
    fig.add_trace(go.Scatter(x=df_continent.Year[df_continent.Continent_Name == "South America"],
                             y=df_continent.AverageTemperature[df_continent.Continent_Name == "South America"],
                             mode="lines+markers", name="América do Sul", text=text[60:72], hoverinfo='text'))

    ########################
    #fig.update_layout(title_text = "Variação da temperatura dos continentes por década entre 1901 e 2013")
    fig.update_layout(
        title=dict(
            text = "Variação da temperatura por continente entre 1901 e 2013",
            font=dict(
                #color="#000000",
                family="arial",
            size = 18
            ),
        x = 0.5),
    # trabalhando a legenda
    legend=dict(
        font=dict(
            #color="#000000",
            family="arial",
            size = 10),
        
        orientation= "h",
        title=dict(
            font=dict(
                #color="#000000",
                family="arial")),
        x=0.04, # move a legenda horizontalmente
        y=-0.2 # move a legenda verticalmente
    ),
    
    font=dict( # fonte geral do gráfico
        #color="#000000",
        family="arial",
        size=14
    ),
    
    separators=",", # separador dos decimais nas palhetas
        
    margin=dict( # altera as margens do gráfico
        autoexpand=False,
        r=50,
        l=70,
        t=100,
        b=130
    ),
    # alteração do fundo do gráfico
    plot_bgcolor="RGB(255, 255, 255)",
    
    # alteração das cores das linhas do gráfico
    colorway=["#ff7f0e", "#f0e516", "#4e51e6", "#d60b30", "#0f5410", "#710e9c"], # cores dos gráficos
    
    # alterações nos eixos x e y
    xaxis=dict(
            showline=True,
            showticklabels=True,
            linecolor='rgb(204, 204, 204)',
            linewidth=1,
            ticks='outside',
            title_text = "Décadas",
            title_font = {"size": 15},
            title_standoff = 60),
            #rangeslider = dict(visible = True)
    yaxis=dict(
            showline=True,
            gridcolor='#ebebf0',
            title_text = "Temperatura em °C",
            title_font = {"size": 15}),
    
    #hovermode = "closest"
    )
    fig.update_yaxes(
        range=[5,25],  # sets the range of xaxis
        #constrain="domain"
    )
    st.plotly_chart(fig, use_container_width=True)

continentes(df_city)


# charts para a variação da temperatura
st.write("**Variações das temperaturas médias por continente** (Cálculos a partir das temperaturas médias nas cidades em cada continente entre 1901 e 2013).")

col1, col2, col3 = st.columns(3)
col4, col5, col6 = st.columns(3)

def variacao_temp(df, continent):
    valor_min = df[df.Continent_Name == continent]["AverageTemperature"].min()
    valor_max = df[df.Continent_Name == continent]["AverageTemperature"].max()
    return valor_min, valor_max

# África
v_min, v_max = variacao_temp(df_city, "Africa")
col1.metric('África', v_max, round((v_max-v_min), 1))

# América do Sul
v_min, v_max = variacao_temp(df_city, "South America")
col2.metric('América do Sul', v_max, round((v_max-v_min), 1))

# Ásia
v_min, v_max = variacao_temp(df_city, "Asia")
col3.metric('Ásia', v_max, round((v_max-v_min), 1))

# América do Norte
v_min, v_max = variacao_temp(df_city, "North America")
col4.metric('América do Norte', v_max, round((v_max-v_min), 1))

# Oceania
v_min, v_max = variacao_temp(df_city, "Oceania")
col5.metric('Oceania', v_max, round((v_max-v_min), 1))

# Europa
v_min, v_max = variacao_temp(df_city, "Europe")
col6.metric('Europa', v_max, round((v_max-v_min), 1))

##############################################################################################################

# Temperaturas por países de escolha
st.subheader("Comparativo entre países / cidades de interesse")
by_countries = read_dataframe("https://raw.githubusercontent.com/dfabat/Projects/main/Dasboard/research_countries.csv")
by_cities = read_dataframe("https://raw.githubusercontent.com/dfabat/Projects/main/Dasboard/by_cities.csv")

is_countries = st.radio("Deseja comparar países ou cidades entre si?", ("Países", "Cidades"))

if is_countries == "Países":
    countries = st.multiselect('Selecione o(s) país(es)', by_countries.Country.unique().tolist(), ['Brazil', 'Denmark'])

    marcar_tudo = st.checkbox('Selecionar todos os países da lista? (Não recomendado)')

    def research_by_country(df, countries, marcar_tudo):
        if marcar_tudo:
            df = df[df.Country.isin(df.Country.unique().tolist())]
            fig = px.line(
                data_frame=df,
                x = "Year",
                y = "AverageTemperature",
                color = "Country",
                markers = True,
                title = "Evolução da temperatura médias nos países selecionados",
                labels={"AverageTemperature": f"Temperatura média em °C",
                        "Year": "Década",
                        "Country": "País"},
                )
            st.plotly_chart(fig, use_container_width=True)
        else:
            df = df[df.Country.isin(countries)]
            fig = px.line(
                data_frame=df,
                x = "Year",
                y = "AverageTemperature",
                color = "Country",
                markers = True,
                title = "Evolução da temperatura médias nos países selecionados",
                labels={"AverageTemperature": f"Temperatura média em °C",
                        "Year": "Década",
                        "Country": "País"})

            st.plotly_chart(fig, use_container_width=True)

    # chamando a função para exibição
    if st.button("Pronto"):
        research_by_country(by_countries, countries, marcar_tudo)

else:
    cities = st.multiselect('Selecione a(s) cidade(s). Digite "País, cidade"', by_cities.City_Country.unique().tolist(), ['Afghanistan, Baglan', 'Brazil, São Paulo'])
        
    def research_by_city(df, cities):
        countries = [string.split(",")[0].strip() for string in cities]
        cities = [string.split(",")[1].strip() for string in cities]
        df = df[(df.Country.isin(countries)) & (df.City.isin(cities))]
        fig = px.line(
            data_frame=df,
            x = "Year",
            y = "AverageTemperature",
            color = "City",
            markers = True,
            title = "Evolução da temperatura médias nas cidades selecionadas",
            labels={"AverageTemperature": "Temperatura média em °C",
                    "Year": "Década",
                    "City": "Cidade"})

        st.plotly_chart(fig, use_container_width=True)

    # chamando a função para exibição
    if st.button("Pronto"):
        research_by_city(by_cities, cities)
    
    
##############################################################################################################

st.subheader("Predição da variação da temperatura nos próximos anos")
periodo = st.slider('Escolha quantos anos serão utilizados na previsão:', 0, 100, 10)    

def predicao(df, periodo):
    df = df[['dt', 'LandAndOceanMovingAverageTemperature12']]
    df.rename(columns={'dt': 'ds', 'LandAndOceanMovingAverageTemperature12': 'y'}, inplace = True)
    m = Prophet()
    m.fit(df)
    future = m.make_future_dataframe(periods=(periodo * 365))
    forecast = m.predict(future)
    return m, forecast

if st.button("Vai"):
    m, forecast = predicao(df_ocean, periodo)
    fig_1 = plot_plotly(m, forecast, xlabel="Anos", ylabel="Temperatura média em °C")
    st.plotly_chart(fig_1, use_container_width=True)

    st.write("**Métricas da predição**")
    fig_2 = m.plot_components(forecast)
    st.write(fig_2)

