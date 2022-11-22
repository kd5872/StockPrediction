
import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objects


@st.cache(allow_output_mutation=True)
def load_data(ticker, start, finish):
    data = yf.download(ticker, start, finish)
    data.reset_index(inplace=True)
    return data


def plot_raw_data(data):
    fig = graph_objects.Figure()
    fig.add_trace(graph_objects.Scatter(x=data['Date'],
                  y=data['Open'], name='stock_open'))
    fig.add_trace(graph_objects.Scatter(x=data['Date'],
                  y=data['Close'], name='stock_close'))
    fig.layout.update(title_text='Time Series Data',
                      xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)


def main():
    START = '2002-01-01'
    TODAY = date.today().strftime('%Y-%m-%d')
    title = '📈Stock Prediction App'
    st.set_page_config(page_title=title)
    st.title(title)
    stocks = ('ICICIBANK.NS', 'AXISBANK.NS', 'HDFCBANK.NS', 'INDUSINDBK.NS')
    selected_stocks = st.selectbox('Select Stock Ticker for prediction', stocks)

    data = load_data(selected_stocks, START, TODAY)

    st.markdown(f'## Data for "{selected_stocks}"')
    st.markdown('###  Raw Data')
    st.write(data.tail())
    st.markdown('### Historical plot')
    plot_raw_data(data)

    st.markdown(f'## Forecast data for "{selected_stocks}"')
    n_years = st.slider('Years of prediction:', 1, 5)
    period = n_years * 365
    data['Date'] = data['Date'].dt.tz_localize(None)
    df_train = data[['Date', 'Close']]
    df_train = df_train.rename(columns={'Date': 'ds', 'Close': 'y'})
    m = Prophet()
    m.fit(df_train)
    future = m.make_future_dataframe(periods=period)
    forecast = m.predict(future)

    st.markdown('### Forecast row')
    st.write(forecast.tail())

    st.markdown('### Forecast plot')
    fig1 = plot_plotly(m, forecast)
    st.plotly_chart(fig1)

    st.markdown('### Forecast components')
    fig2 = m.plot_components(forecast)
    st.write(fig2)


if __name__ == '__main__':
    main()