import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import base64
import io

from dash import Dash, dcc, html, dash_table, Input, Output
import dash_cytoscape as cyto

from tqdm import tqdm

tqdm.pandas()
import time  # Для вимірювання часу виконання

# ----------------------------------------------------------
# 1) ІНІЦІАЛІЗАЦІЯ ДАНИХ (ЗАМІНІТЬ НА ВАШЕ ЗАВАНТАЖЕННЯ CSV)
# ----------------------------------------------------------

# Припустимо, що у вас є DataFrame fact_trans,
# але в цьому прикладі штучно згенеруємо дані.
np.random.seed(42)

REGIONS = ["Region A", "Region B", "Region C", "Region D"]
YEARS = [1993, 1994, 1995, 1996, 1997, 1998]
CLIENT_IDS = list(range(1, 21))

fact_trans = pd.DataFrame({
    "trans_id": range(1, 101),
    "year": np.random.choice(YEARS, 100),
    "region": np.random.choice(REGIONS, 100),
    "trans_amount": np.random.randint(100, 5000, 100),
    "loan_amount": np.random.randint(0, 3000, 100),  # Припустимо, це позначає суму кредиту
    "client_id": np.random.choice(CLIENT_IDS, 100),
    # Додамо умовний стовпець type_of_trans, який міг би зустрічатися в The Berka Dataset
    "type_of_trans": np.random.choice(["withdrawal", "deposit", "credit_card_payment", "loan_repayment"], 100),
})

# Створимо fact_loan для додаткових KPI
fact_loan = pd.DataFrame({
    "loan_id": range(1, 31),
    "client_id": np.random.choice(CLIENT_IDS, 30),
    "loan_type": np.random.choice(["mortgage", "consumer", "credit_card"], 30),
    "loan_balance": np.random.randint(1000, 10000, 30),
})

# Створимо dim_client зі статичними даними
dim_client = pd.DataFrame({
    "client_id": CLIENT_IDS,
    "district_id": np.random.randint(1, 5, len(CLIENT_IDS)),  # умовно
    "gender": np.random.choice(["M", "F"], len(CLIENT_IDS)),
    "age": np.random.randint(18, 70, len(CLIENT_IDS)),
})

# ----------------------------------------------------------
# 2) ОБЧИСЛЕННЯ KPI (10)
# ----------------------------------------------------------

# 1) Total Transactions
total_transactions = len(fact_trans)

# 2) Average Transaction Amount
avg_trans_amount = fact_trans["trans_amount"].mean()

# 3) Total Loan Amount (сума loan_amount у fact_trans)
total_loan_amount = fact_trans["loan_amount"].sum()

# 4) Unique Clients
unique_clients = fact_trans["client_id"].nunique()

# 5) Mortgage Loans Count (у fact_loan, умовно)
mortgage_count = len(fact_loan[fact_loan["loan_type"] == "mortgage"])

# 6) Total Loan Balance
total_loan_balance = fact_loan["loan_balance"].sum()

# 7) Average Loan Balance per Client
avg_loan_balance_per_client = total_loan_balance / unique_clients if unique_clients != 0 else 0

# 8) Loan to Transaction Ratio
loan_to_trans_ratio = total_loan_amount / total_transactions if total_transactions != 0 else 0

# 9) Transaction Growth Rate by Year
df_trans_per_year = fact_trans.groupby("year")["trans_id"].count().reset_index(name='trans_count').sort_values('year')
df_trans_per_year['growth_rate'] = df_trans_per_year['trans_count'].pct_change() * 100

# 10) Average Transaction Amount by Type
avg_trans_by_type = fact_trans.groupby("type_of_trans")["trans_amount"].mean().reset_index()

# ----------------------------------------------------------
# 3) ПІДГОТОВКА ДАНИХ ДЛЯ ГРАФІКІВ (9 РІЗНИХ ТИПІВ)
# ----------------------------------------------------------

# Bar Chart: сума транзакцій за регіонами
df_region = fact_trans.groupby("region")["trans_amount"].sum().reset_index()

# Line Chart: сума транзакцій по роках
df_year = fact_trans.groupby("year")["trans_amount"].sum().reset_index().sort_values("year")

# Pie Chart: розподіл типів транзакцій
df_type = fact_trans.groupby("type_of_trans")["trans_amount"].sum().reset_index()

# Scatter Chart: зв'язок між trans_amount та loan_amount
df_scatter = fact_trans[["trans_id", "trans_amount", "loan_amount", "region"]]

# Donut Chart: розподіл клієнтів за статтю
df_gender = dim_client.groupby("gender")["client_id"].count().reset_index()

# Histogram of Transaction Amounts
df_hist = fact_trans["trans_amount"]

# Box Plot of Loan Balances by Loan Type
df_box = fact_loan

# Heatmap of Transactions by Region and Year
df_heatmap = fact_trans.groupby(["region", "year"])["trans_amount"].sum().reset_index()
df_heatmap_pivot = df_heatmap.pivot(index='region', columns='year', values='trans_amount').fillna(0)

# Age Distribution of Clients
df_age = dim_client["age"]


# ----------------------------------------------------------
# 4) ТАБЛИЦЯ ЗІ СПАРКЛАЙНАМИ (SPARKLINES)
# ----------------------------------------------------------

def generate_svg_sparkline(data, width=100, height=30, color='blue'):
    """
    Генерує base64-зображення SVG графіка (Sparkline) з масиву даних.
    """
    max_val = max(data) if max(data) != 0 else 1
    points = " ".join([f"{i},{height - (d / max_val * height)}" for i, d in enumerate(data)])
    svg = f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">' \
          f'<polyline points="{points}" style="fill:none;stroke:{color};stroke-width:1" />' \
          f'</svg>'
    encoded = base64.b64encode(svg.encode('utf-8')).decode('utf-8')
    return f'data:image/svg+xml;base64,{encoded}'


# Обмежимо кількість спарклайнів до топ-10 регіонів за транзакціями
df_spark_example = df_region.nlargest(10, 'trans_amount').copy()

# Генеруємо spark_data
df_spark_example["spark_data"] = [
    fact_trans[fact_trans["region"] == region]["trans_amount"].rolling(window=12, min_periods=1).sum().tolist()
    for region in df_spark_example["region"]
]

# Генеруємо спарклайни з прогрес-баром
start_time = time.time()
df_spark_example["sparkline"] = df_spark_example["spark_data"].progress_apply(
    lambda x: generate_svg_sparkline(x[:12])  # обмежимо до 12 точок для швидкості
)
end_time = time.time()

print(f"Sparkline generation completed in {end_time - start_time:.2f} seconds.")

# ----------------------------------------------------------
# 5) СТВОРЕННЯ DASH-ДОДАТКУ
# ----------------------------------------------------------
app = Dash(__name__)
app.title = "Czech Bank Extended BI Dashboard"

# ----------------------------------------------------------
# 6) LAYOUT І СТРУКТУРА DASH-ДОДАТКУ
# ----------------------------------------------------------
app.layout = html.Div([
    html.H1("Extended BI Dashboard", style={'textAlign': 'center'}),
    html.Hr(),

    # KPI Cards (10 шт)
    html.Div([
        html.Div([
            html.H3("Total Transactions"),
            html.H2(f"{total_transactions}", id='kpi-total-transactions', style={'color': 'blue'})
        ], style={'border': '2px solid #ddd', 'padding': '10px', 'margin': '10px', 'width': '18%',
                  'textAlign': 'center'}),

        html.Div([
            html.H3("Avg Transaction Amount"),
            html.H2(f"{avg_trans_amount:.2f}", id='kpi-avg-trans-amount', style={'color': 'blue'})
        ], style={'border': '2px solid #ddd', 'padding': '10px', 'margin': '10px', 'width': '18%',
                  'textAlign': 'center'}),

        html.Div([
            html.H3("Total Loan Amount"),
            html.H2(f"{total_loan_amount}", id='kpi-total-loan-amount', style={'color': 'blue'})
        ], style={'border': '2px solid #ddd', 'padding': '10px', 'margin': '10px', 'width': '18%',
                  'textAlign': 'center'}),

        html.Div([
            html.H3("Unique Clients"),
            html.H2(f"{unique_clients}", id='kpi-unique-clients', style={'color': 'blue'})
        ], style={'border': '2px solid #ddd', 'padding': '10px', 'margin': '10px', 'width': '18%',
                  'textAlign': 'center'}),

        html.Div([
            html.H3("Mortgage Count"),
            html.H2(f"{mortgage_count}", id='kpi-mortgage-count', style={'color': 'blue'})
        ], style={'border': '2px solid #ddd', 'padding': '10px', 'margin': '10px', 'width': '18%',
                  'textAlign': 'center'}),

        html.Div([
            html.H3("Total Loan Balance"),
            html.H2(f"{total_loan_balance}", id='kpi-total-loan-balance', style={'color': 'blue'})
        ], style={'border': '2px solid #ddd', 'padding': '10px', 'margin': '10px', 'width': '18%',
                  'textAlign': 'center'}),

        html.Div([
            html.H3("Avg Loan Balance per Client"),
            html.H2(f"{avg_loan_balance_per_client:.2f}", id='kpi-avg-loan-balance', style={'color': 'blue'})
        ], style={'border': '2px solid #ddd', 'padding': '10px', 'margin': '10px', 'width': '18%',
                  'textAlign': 'center'}),

        html.Div([
            html.H3("Loan to Transaction Ratio"),
            html.H2(f"{loan_to_trans_ratio:.2f}", id='kpi-loan-trans-ratio', style={'color': 'blue'})
        ], style={'border': '2px solid #ddd', 'padding': '10px', 'margin': '10px', 'width': '18%',
                  'textAlign': 'center'}),

        html.Div([
            html.H3("Transaction Growth Rate"),
            dash_table.DataTable(
                id='kpi-trans-growth-rate',
                columns=[{"name": "Year", "id": "year"}, {"name": "Growth Rate (%)", "id": "growth_rate"}],
                data=df_trans_per_year.dropna().to_dict('records'),
                style_cell={'textAlign': 'center'},
                style_header={'backgroundColor': '#f1f1f1', 'fontWeight': 'bold'},
                style_table={'width': '100%'},
                page_size=5
            )
        ], style={'border': '2px solid #ddd', 'padding': '10px', 'margin': '10px', 'width': '35%',
                  'textAlign': 'center'}),

        html.Div([
            html.H3("Avg Transaction Amount by Type"),
            dash_table.DataTable(
                id='kpi-avg-trans-by-type',
                columns=[{"name": "Transaction Type", "id": "type_of_trans"},
                         {"name": "Avg Amount", "id": "trans_amount"}],
                data=avg_trans_by_type.to_dict('records'),
                style_cell={'textAlign': 'center'},
                style_header={'backgroundColor': '#f1f1f1', 'fontWeight': 'bold'},
                style_table={'width': '50%'},
                page_size=4
            )
        ], style={'border': '2px solid #ddd', 'padding': '10px', 'margin': '10px', 'width': '35%',
                  'textAlign': 'center'}),
    ], style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'center'}),

    html.Hr(),

    # ФІЛЬТР (Dropdown) для регіону та віку
    html.Div([
        html.Label("Оберіть Регіон:"),
        dcc.Dropdown(
            id='region-filter',
            options=[{'label': r, 'value': r} for r in df_region["region"].unique()],
            value=None,  # початково None -> усі регіони
            clearable=True,
            placeholder="Всі регіони"
        )
    ], style={'width': '30%', 'margin': '10px', 'display': 'inline-block'}),

    html.Div([
        html.Label("Оберіть Вік Клієнтів:"),
        dcc.RangeSlider(
            id='age-filter',
            min=dim_client['age'].min(),
            max=dim_client['age'].max(),
            step=1,
            value=[dim_client['age'].min(), dim_client['age'].max()],
            marks={i: str(i) for i in range(dim_client['age'].min(), dim_client['age'].max() + 1, 10)},
            tooltip={"placement": "bottom", "always_visible": True}
        )
    ], style={'width': '50%', 'margin': '20px'}),

    # 9 Графіків
    html.Div([
        # 1) Bar Chart (Сума транзакцій за регіонами, з фільтром)
        html.Div([
            html.H3("Bar Chart: Transactions by Region"),
            dcc.Graph(id='bar-chart-region')
        ], style={'width': '48%', 'display': 'inline-block'}),

        # 2) Line Chart: Сума транзакцій по роках
        html.Div([
            html.H3("Line Chart: Transactions by Year"),
            dcc.Graph(
                figure=px.line(
                    df_year,
                    x='year',
                    y='trans_amount',
                    title="Sum of Transactions by Year",
                    template='plotly_white'
                )
            )
        ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),
    ], style={'display': 'flex', 'justify-content': 'space-around'}),

    html.Div([
        # 3) Pie Chart: Розподіл типів транзакцій
        html.Div([
            html.H3("Pie Chart: Type of Transactions"),
            dcc.Graph(
                figure=px.pie(
                    df_type,
                    names='type_of_trans',
                    values='trans_amount',
                    title="Transaction Types Distribution",
                    template='plotly_white'
                )
            )
        ], style={'width': '48%', 'display': 'inline-block'}),

        # 4) Scatter Chart: trans_amount vs loan_amount
        html.Div([
            html.H3("Scatter Chart: Trans vs. Loan"),
            dcc.Graph(
                figure=px.scatter(
                    df_scatter,
                    x='trans_amount',
                    y='loan_amount',
                    color='region',
                    title="Transaction Amount vs. Loan Amount",
                    template='plotly_white',
                    hover_data=['trans_id']  # Переконайтеся, що 'trans_id' присутній у df_scatter
                )
            )
        ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),
    ], style={'display': 'flex', 'justify-content': 'space-around'}),

    html.Div([
        # 5) Donut Chart: Розподіл клієнтів за статтю
        html.Div([
            html.H3("Donut Chart: Clients by Gender"),
            dcc.Graph(
                figure=px.pie(
                    df_gender,
                    names='gender',
                    values='client_id',
                    hole=0.4,
                    title="Clients by Gender (Donut)",
                    template='plotly_white'
                )
            )
        ], style={'width': '48%', 'display': 'inline-block'}),

        # 6) Histogram of Transaction Amounts
        html.Div([
            html.H3("Histogram: Transaction Amounts"),
            dcc.Graph(
                figure=px.histogram(
                    df_hist,
                    x='trans_amount',
                    nbins=20,
                    title="Distribution of Transaction Amounts",
                    template='plotly_white'
                )
            )
        ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),
    ], style={'display': 'flex', 'justify-content': 'space-around'}),

    html.Div([
        # 7) Box Plot of Loan Balances by Loan Type
        html.Div([
            html.H3("Box Plot: Loan Balances by Type"),
            dcc.Graph(
                figure=px.box(
                    df_box,
                    x='loan_type',
                    y='loan_balance',
                    title="Loan Balances Distribution by Loan Type",
                    template='plotly_white'
                )
            )
        ], style={'width': '48%', 'display': 'inline-block'}),

        # 8) Heatmap of Transactions by Region and Year
        html.Div([
            html.H3("Heatmap: Transactions by Region and Year"),
            dcc.Graph(
                figure=px.imshow(
                    df_heatmap_pivot,
                    labels=dict(x="Year", y="Region", color="Transaction Amount"),
                    x=df_heatmap_pivot.columns,
                    y=df_heatmap_pivot.index,
                    title="Heatmap of Transactions by Region and Year",
                    color_continuous_scale='Blues'
                )
            )
        ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),
    ], style={'display': 'flex', 'justify-content': 'space-around'}),

    html.Div([
        # 9) Age Distribution of Clients
        html.Div([
            html.H3("Histogram: Age Distribution of Clients"),
            dcc.Graph(
                figure=px.histogram(
                    df_age,
                    x='age',
                    nbins=10,
                    title="Age Distribution of Clients",
                    template='plotly_white'
                )
            )
        ], style={'width': '48%', 'display': 'inline-block'}),
    ], style={'display': 'flex', 'justify-content': 'center'}),

    html.Hr(),

    # Таблиця зі спарклайнами
    html.H2("Table with Sparklines"),
    dash_table.DataTable(
        id='sparkline-table',
        columns=[
            {"name": "Region", "id": "region"},
            {"name": "Transaction Sum", "id": "trans_amount"},
            {"name": "Sparkline", "id": "sparkline", "presentation": "markdown"}
        ],
        data=[{
            "region": row["region"],
            "trans_amount": row["trans_amount"],
            "sparkline": f"<img src='{row['sparkline']}' style='width:100px; height:30px;' />"
        } for _, row in df_spark_example.iterrows()],
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'center'},
        markdown_options={"html": True},
        page_size=10,
    ),

    html.Hr(),

    # Відображення метаданих (можна розширити)
    html.H2("Метадані"),
    html.Div([
        html.H4("Dimension Tables:"),
        html.Ul([
            html.Li("dim_client: client_id, district_id, gender, age"),
            # Додайте інші Dimension Tables за потреби
        ]),
        html.H4("Fact Tables:"),
        html.Ul([
            html.Li("fact_trans: trans_id, year, region, trans_amount, loan_amount, client_id, type_of_trans"),
            html.Li("fact_loan: loan_id, client_id, loan_type, loan_balance"),
            # Додайте інші Fact Tables за потреби
        ])
    ], style={'padding': '20px'})
])


# ----------------------------------------------------------
# 7) CALLBACK: ФІЛЬТРУВАННЯ ГРАФІКІВ ЗА РЕГІОНОМ ТА ВІКОМ
# ----------------------------------------------------------
@app.callback(
    [
        Output('bar-chart-region', 'figure'),
        Output('kpi-total-transactions', 'children'),
        Output('kpi-avg-trans-amount', 'children'),
        Output('kpi-total-loan-amount', 'children'),
        Output('kpi-unique-clients', 'children'),
        Output('kpi-mortgage-count', 'children'),
        Output('kpi-total-loan-balance', 'children'),
        Output('kpi-avg-loan-balance', 'children'),
        Output('kpi-loan-trans-ratio', 'children'),
        Output('kpi-trans-growth-rate', 'data'),
        Output('kpi-avg-trans-by-type', 'data'),
    ],
    [
        Input('region-filter', 'value'),
        Input('age-filter', 'value')
    ]
)
def update_dashboard(selected_region, age_range):
    """
    Оновлює графіки та KPI залежно від вибраних фільтрів (регіон та вік).
    """
    # Фільтрація даних за регіоном та віком
    if selected_region:
        filtered_trans = fact_trans[
            (fact_trans["region"] == selected_region) &
            (fact_trans["client_id"].isin(dim_client[
                                              (dim_client['age'] >= age_range[0]) &
                                              (dim_client['age'] <= age_range[1])
                                              ]['client_id']))
            ]
    else:
        filtered_trans = fact_trans[
            fact_trans["client_id"].isin(dim_client[
                                             (dim_client['age'] >= age_range[0]) &
                                             (dim_client['age'] <= age_range[1])
                                             ]['client_id'])
        ]

    # Оновлення Bar Chart
    df_bar = filtered_trans.groupby("region")["trans_amount"].sum().reset_index()
    fig_bar = px.bar(
        df_bar,
        x='region',
        y='trans_amount',
        title="Sum of Transactions by Region (Filtered)" if selected_region else "Sum of Transactions by Region",
        template='plotly_white'
    )

    # Оновлення KPI
    total_trans = len(filtered_trans)
    avg_trans = filtered_trans["trans_amount"].mean() if total_trans != 0 else 0
    total_loan = filtered_trans["loan_amount"].sum()
    unique_cl = filtered_trans["client_id"].nunique()

    # Фільтрація loans
    filtered_loans = fact_loan[fact_loan["client_id"].isin(filtered_trans["client_id"])]
    mort_count = len(filtered_loans[filtered_loans["loan_type"] == "mortgage"])
    tot_loan_bal = filtered_loans["loan_balance"].sum()

    # Середній баланс кредиту на клієнта
    avg_loan_bal = tot_loan_bal / unique_cl if unique_cl != 0 else 0

    # Loan to Transaction Ratio
    loan_trans_ratio = total_loan / total_trans if total_trans != 0 else 0

    # Transaction Growth Rate (за фільтрованими даними)
    df_trans_filtered = filtered_trans.groupby("year")["trans_id"].count().reset_index(name='trans_count').sort_values(
        'year')
    df_trans_filtered['growth_rate'] = df_trans_filtered['trans_count'].pct_change() * 100
    trans_growth = df_trans_filtered.dropna()[['year', 'growth_rate']].to_dict('records')

    # Avg Transaction Amount by Type (за фільтрованими даними)
    avg_trans_type = filtered_trans.groupby("type_of_trans")["trans_amount"].mean().reset_index()
    avg_trans_by_type_filtered = avg_trans_type.to_dict('records')

    return (
        fig_bar,
        f"{total_trans}",
        f"{avg_trans:.2f}" if not np.isnan(avg_trans) else "0",
        f"{total_loan}",
        f"{unique_cl}",
        f"{mort_count}",
        f"{tot_loan_bal}",
        f"{avg_loan_bal:.2f}",
        f"{loan_trans_ratio:.2f}",
        trans_growth,
        avg_trans_by_type_filtered
    )


# ----------------------------------------------------------
# 8) CALLBACK: ОНОВЛЕННЯ Таблиці зі Спарклайнами ЗА ФІЛЬТРАМИ
# ----------------------------------------------------------
@app.callback(
    Output('sparkline-table', 'data'),
    [
        Input('region-filter', 'value'),
        Input('age-filter', 'value')
    ]
)
def update_sparkline_table(selected_region, age_range):
    """
    Оновлює таблицю зі спарклайнами залежно від вибраних фільтрів (регіон та вік).
    """
    # Фільтрація даних за регіоном та віком
    if selected_region:
        filtered = df_spark_example[df_spark_example["region"] == selected_region]
    else:
        filtered = df_spark_example.copy()

    # Додатково фільтруємо за віком клієнтів
    if age_range:
        relevant_clients = dim_client[
            (dim_client['age'] >= age_range[0]) &
            (dim_client['age'] <= age_range[1])
            ]['client_id']
        filtered = filtered[filtered["region"].isin(
            fact_trans[
                fact_trans["client_id"].isin(relevant_clients)
            ]["region"].unique()
        )]

    return [{
        "region": row["region"],
        "trans_amount": row["trans_amount"],
        "sparkline": f"<img src='{row['sparkline']}' style='width:100px; height:30px;' />"
    } for _, row in filtered.iterrows()]


# ----------------------------------------------------------
# 9) Запуск Додатку
# ----------------------------------------------------------
if __name__ == '__main__':
    app.run_server(debug=True, port=8050)
