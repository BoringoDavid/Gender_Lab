import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Must be set before importing pyplot
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
# from dash import dcc, html


# LIST OF INDICATORS-----------------------------------------------------------------------
# LFS INDICATORS-------------------------------------------------------------------------------
    # 1. Distribution of the labour force participation rate by area of residence and sex (in %)

# The function to generate the table
def generate_LFPR_table(df):
    grouped_data = df.groupby(['Code_UR', 'A01'])['LFPR'].mean().reset_index()
    national_data = df.groupby('A01')['LFPR'].mean().reset_index()
    national_data['Code_UR'] = 'National'
    final_data = pd.concat([grouped_data, national_data], ignore_index=True)
    lfpr_table = final_data.pivot(index="Code_UR", columns="A01", values="LFPR")
    lfpr_table.columns.name = "Sex"
    lfpr_table.index.name = "Area of Residence"
    return lfpr_table


# The function to generate the figure
def plot_LFPR(df):
    """
    Plots the Labour Force Participation Rate (LFPR) by area of residence and sex, 
    including a national-level summary.

    Parameters:
        df (pd.DataFrame): DataFrame containing 'Code_UR', 'A01', and 'LFPR' columns.
    """
    # Group by area of residence and sex
    grouped_data = df.groupby(['Code_UR', 'A01'])['LFPR'].mean().reset_index()

    # Compute national-level LFPR
    national_data = df.groupby('A01')['LFPR'].mean().reset_index()
    national_data['Code_UR'] = 'National'  # Label for national-level data

    # Combine all data
    final_data = pd.concat([grouped_data, national_data], ignore_index=True)

    # Set Seaborn style
    sns.set_theme(style="whitegrid")

    # Create the grouped bar plot
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(
        x="Code_UR", y="LFPR", hue="A01", data=final_data,
        palette=sns.color_palette("husl"),  # Beautiful color contrast
        edgecolor="black"
    )

    # Add values on bars
    for p in ax.patches:
        if p.get_height() > 0:  # Only annotate non-zero values
            ax.annotate(
                f"{p.get_height():.1f}%", 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha="center", va="bottom", 
                fontsize=12, fontweight="bold", color="black"
            )

    # Formatting
    plt.title("Labour Force Participation Rate by Area of Residence and Sex", fontsize=14, fontweight="bold")
    plt.xlabel("Area of Residence", fontsize=12)
    plt.ylabel("Labour Force Participation Rate (%)", fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(title="Sex", title_fontsize="12", fontsize="10")

    # Save the plot as a PNG image in a buffer
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    
    # Convert the image to base64 encoding to be used in the Dash app
    img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')

    # Close the plot to avoid display in non-interactive environments
    plt.close()

    return img_base64




    # 2. Distribution of the labour force participation rate by age groups and sex (in %)

# The function for generating table

def LFPR_by_agegroup_sex(df):
    """
    Generates a table showing the Labour Force Participation Rate (LFPR) 
    by age groups and sex.

    Parameters:
        df (pd.DataFrame): DataFrame containing 'age10', 'A01' (Sex), and 'LFPR' columns.

    Returns:
        pd.DataFrame: Aggregated table showing LFPR by age groups and sex.
    """

    # Convert columns to appropriate types
    df['age10'] = df['age10'].astype('category')
    df['LFPR'] = df['LFPR'].astype(float)

    # Handle null values
    df = df.dropna(subset=['age10', 'LFPR'])

    # Compute the mean LFPR for each age group and sex
    table = df.groupby(['age10', 'A01'])['LFPR'].mean().unstack().round(1)

    # Rename columns for clarity
    table.columns.name = None
    table.reset_index(inplace=True)
    table.rename(columns={'age10': 'Age Group'}, inplace=True)

    return table



# The function to generate figure
import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns

def plot_LFPR_by_agegroup_sex(df):
    """
    Plots the Labour Force Participation Rate (LFPR) by age groups and sex,
    and returns a base64-encoded PNG image to be displayed in Dash.

    Parameters:
        df (pd.DataFrame): DataFrame containing 'age10', 'A01' (Sex), and 'LFPR' columns.

    Returns:
        str: Base64 encoded PNG image.
    """
    # Convert columns into their suitable data types
    df['age10'] = df['age10'].astype('category')
    df['LFPR'] = df['LFPR'].astype(float)

    # Handle null values
    df = df.dropna(subset=['age10', 'LFPR'])

    # Set Seaborn Style
    sns.set_theme(style="whitegrid")

    # Create the bar plot
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(
        x="age10", y="LFPR", hue="A01", data=df,
        palette=sns.color_palette("husl"),
        edgecolor="black"
    )

    # Add values on bars
    for p in ax.patches:
        if p.get_height() > 0:  # Only annotate non-zero values
            ax.annotate(
                f"{p.get_height():.1f}%", 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha="center", va="bottom", 
                fontsize=12, fontweight="bold", color="black"
            )

    # Formatting
    plt.title("Labour Force Participation Rate by Age Groups and Sex", fontsize=14, fontweight="bold")
    plt.xlabel("Age Groups", fontsize=12)
    plt.ylabel("Labour Force Participation Rate (%)", fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(title="Sex", title_fontsize="12", fontsize="10")

    # Save the plot as a PNG image in a buffer
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)

    # Convert the image to base64 encoding to be used in the Dash app
    img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')

    # Close the plot to avoid display issues
    plt.close()

    return img_base64









# EICV INDICATORS-------------------------------------------------------------------------------

    # 1. Population structure (%), by sex and five-year age group.

# The function for generating table

def population_structure_by_sex_five_year_group(df):
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    age_bins = list(range(0, 70, 5)) + [100]
    age_labels = [f'{i}-{i+4}' for i in range(0, 65, 5)] + ['65+']
    df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, right=False)

    if 'weight' in df.columns:
        counts = df.groupby(['age_group', 'sex'])['weight'].sum().unstack(fill_value=0)
    else:
        counts = df.groupby(['age_group', 'sex']).size().unstack(fill_value=0)

    percents = counts.div(counts.sum(axis=1), axis=0) * 100

    counts_fmt = counts.applymap(lambda x: f"{x:,.0f}")
    percents_fmt = percents.applymap(lambda x: f"{x:.1f}%")

    counts_fmt.columns.name = percents_fmt.columns.name = 'age_group'
    counts_fmt.index.name = percents_fmt.index.name = 'sex'

    table = pd.concat([counts_fmt, percents_fmt], axis=1, keys=["Count", "Percentage"])

    # 👇 FLATTEN THE MULTIINDEX
    table.columns = ['_'.join(col).strip() for col in table.columns.values]
    table.reset_index(inplace=True)

    return table


# The function for generating figure

import pandas as pd
import matplotlib.pyplot as plt
import io
import base64

# function for column renaming
def df_col_rename(df):
    df.rename(columns={
        "ur_x": "ur",
        "s1q4": "maritalstatus",
        "s1q3y": "age",
        "s1q2": "relation",
        "weight_x": "weight",
        "province_x": "province",
        "s1q1": "sex",
        "s5aq1": "type_of_habitat",
        "s5aq11": "occupancy_status",
        "s5dq2": "main_roofing_material",
        "s5dq1": "main_construction_material",
        "s5dq3": "main_floor_material",
        "s5cq18": "main_cooking_fuel",
        "s5cq16": "main_source_of_lighting",
        "s5cq26": "households_with_internet_access",
        "s5cq1": "drinking_water_source",
        "s5cq20": "improved_sanitation",
        "poverty_x": "poverty_level"
    }, inplace=True)
    return df

# Main function
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64

def plot_population_structure_by_sex_five_year_group(df, plot=True):
    df = df_col_rename(df)
    df['age'] = pd.to_numeric(df['age'], errors='coerce')

    # Define age bins and labels
    age_bins = list(range(0, 70, 5)) + [100]
    age_labels = [f'{i}-{i+4}' for i in range(0, 65, 5)] + ['65+']
    df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, right=False)

    # Aggregate by age group and sex
    if 'weight' in df.columns:
        counts = df.groupby(['age_group', 'sex'])['weight'].sum().unstack(fill_value=0)
    else:
        counts = df.groupby(['age_group', 'sex']).size().unstack(fill_value=0)

    # Ensure both 'Male' and 'Female' columns exist
    if 'Male' not in counts.columns:
        counts['Male'] = 0
    if 'Female' not in counts.columns:
        counts['Female'] = 0

    # Add totals
    counts['Total'] = counts[['Male', 'Female']].sum(axis=1)
    grand_total = counts['Total'].sum()

    # Percentages
    row_totals = counts[['Male', 'Female']].sum(axis=1)
    percents = counts[['Male', 'Female']].div(row_totals, axis=0) * 100

    # Append total row
    total_row_counts = counts[['Male', 'Female', 'Total']].sum().to_frame().T
    total_row_counts.index = ['Total']
    total_row_percents = total_row_counts[['Male', 'Female']].div(grand_total) * 100

    counts = pd.concat([counts, total_row_counts])
    percents = pd.concat([percents, total_row_percents])

    # Format for display
    counts_fmt = counts.applymap(lambda x: f"{x:,.0f}")
    percents_fmt = percents.applymap(lambda x: f"{x:.1f}%")

    counts_fmt.columns.name = percents_fmt.columns.name = 'sex'
    counts_fmt.index.name = percents_fmt.index.name = 'age_group'

    result = pd.concat([counts_fmt, percents_fmt], axis=1, keys=["Count", "Percentage"])

    # Plot pyramid
    img_base64 = None
    if plot:
        counts_df = counts.loc[age_labels]

        male = -counts_df['Male']
        female = counts_df['Female']
        total_population = female.sum() + (-male).sum()
        age_groups = counts_df.index.astype(str)

        male_percents = (-male / total_population * 100).round(1)
        female_percents = (female / total_population * 100).round(1)

        fig, ax = plt.subplots(figsize=(10, 7))

        bars_male = ax.barh(age_groups, male, color='steelblue', label='Male')
        bars_female = ax.barh(age_groups, female, color='lightcoral', label='Female')

        # Add % labels to male bars (left side)
        for bar, percent in zip(bars_male, male_percents):
            ax.text(bar.get_width() - 500,
                    bar.get_y() + bar.get_height() / 2,
                    f"{percent:.1f}%",
                    ha='right', va='center', color='black', fontsize=9)

        # Add % labels to female bars (right side)
        for bar, percent in zip(bars_female, female_percents):
            ax.text(bar.get_width() + 500,
                    bar.get_y() + bar.get_height() / 2,
                    f"{percent:.1f}%",
                    ha='left', va='center', color='black', fontsize=9)

        ax.set_xlabel("Population")
        ax.set_ylabel("Age Group")
        ax.set_title("Population Pyramid by Sex and Age Group")
        ax.legend()

        # Format x-axis to show absolute values
        xticks = ax.get_xticks()
        ax.set_xticklabels([f'{abs(int(x)):,}' for x in xticks])

        plt.tight_layout()

        # Save the figure into a BytesIO buffer
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)

        # Encode the image to base64
        img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')

        # Close plot
        plt.close()

    return fig, img_base64


# 2. ========================================== Number of Females per 100 Males by Province and Age Group ======================================================
import numpy as np
def females_per_100_males(df):
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    age_bins = list(range(0, 70, 5)) + [100]
    age_labels = [f'{i}-{i+4}' for i in range(0, 65, 5)] + ['65+']
    df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, right=False)
    
    data = df.groupby(['province', 'age_group', 'sex']).size().unstack(fill_value=0)
    if 'Male' in data.columns and 'Female' in data.columns:
        data['females_per_100_males'] = (data['Female'] / data['Male']) * 100
    else:
        data['females_per_100_males'] = np.nan

    for col in ['Male', 'Female']:
        if col in data.columns:
            data[col] = data[col].map(lambda x: f"{x:,}")
    data['females_per_100_males'] = data['females_per_100_males'].map(
        lambda x: f"{x:.1f}%" if pd.notnull(x) else "N/A"
    )

     # 👇 FLATTEN THE MULTIINDEX
    table = data
    table.columns = ['_'.join(col).strip() for col in table.columns.values]
    table.reset_index(inplace=True)
    return table
# 2a. ================================================Function: Sex of Household Heads by Province===========================================================
def sex_of_household_heads_tbl(df):
    hh = df[df['relation'] == 'Household head _HH_'].copy()

    result = hh.groupby(['province', 'sex']).size().unstack(fill_value=0)
    result['Total'] = result.sum(axis=1)
    result['% Headed by Male'] = (result.get('Male', 0) / result['Total']) * 100
    result['% Headed by Female'] = (result.get('Female', 0) / result['Total']) * 100
    result['% of Total'] = (result['Total'] / result['Total'].sum()) * 100

    for col in ['Male', 'Female', 'Total']:
        if col in result.columns:
            result[col] = result[col].apply(lambda x: f"{x:,}")

    for col in ['% Headed by Male', '% Headed by Female', '% of Total']:
        result[col] = result[col].apply(lambda x: f"{x:.1f}%")

    result.index.name = 'Province'
    result.reset_index(inplace=True)
    result.columns.name = 'Sex of HH Head'

    return result

# 2a. Function: Sex of Household Heads by Province
def sex_of_household_heads_plot(df):
    hh = df[df['relation'] == 'Household head _HH_'].copy()

    result = hh.groupby(['province', 'sex']).size().unstack(fill_value=0)
    result['Total'] = result.sum(axis=1)
    result['% Headed by Male'] = (result.get('Male', 0) / result['Total']) * 100
    result['% Headed by Female'] = (result.get('Female', 0) / result['Total']) * 100
    result['% of Total'] = (result['Total'] / result['Total'].sum()) * 100

    for col in ['Male', 'Female', 'Total']:
        if col in result.columns:
            result[col] = result[col].apply(lambda x: f"{x:,}")

    for col in ['% Headed by Male', '% Headed by Female', '% of Total']:
        result[col] = result[col].apply(lambda x: f"{x:.1f}%")

    result.index.name = 'Province'
    result.columns.name = 'Sex of HH Head'

    # Pie chart of total female vs male headed households
    total_counts = hh['sex'].value_counts()
    if set(['Male', 'Female']).issubset(total_counts.index):
        total_counts = total_counts[['Male', 'Female']]

    fig = plt.figure(figsize=(6, 4))
    wedges, texts, autotexts = plt.pie(
        total_counts,
        labels=total_counts.index,
        autopct='%1.1f%%',
        colors=['#4C72B0', '#DD8452'],
        startangle=140,
        labeldistance=0.8
    )

    for text in texts:
        text.set_horizontalalignment('left')

    plt.title('Distribution of Household Heads by Sex (National)')
    plt.axis('equal')
    plt.tight_layout()
    # Save the figure into a BytesIO buffer
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)

    # Encode the image to base64
    img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')

    # Close plot
    plt.close()

    return fig, img_base64

# 5a. =====================================================Function: Poverty Levels by Sex with Chart ====================================================
def poverty_level_by_sex_plot(merged_df):
    # Weighted or unweighted counts
    if 'weight' in merged_df.columns:
        grouped = merged_df.groupby(['province', 'ur', 'sex', 'poverty_level'])['weight'].sum()
    else:
        grouped = merged_df.groupby(['province', 'ur', 'sex', 'poverty_level']).size()

    poverty_counts = grouped.unstack(fill_value=0)
    row_totals = poverty_counts.sum(axis=1)
    poverty_percent = poverty_counts.divide(row_totals, axis=0) * 100

    # Combine counts and percentages into one cell
    pov_combined = poverty_counts.copy()
    for col in poverty_counts.columns:
        pov_combined[col] = poverty_counts[col].astype(int).map(lambda x: f"{x:,.0f}") + \
                        ' (' + poverty_percent[col].map(lambda x: f"{x:.1f}%") + ')'

    pov_combined['Total'] = row_totals.map(lambda x: f"{x:,.0f}")
    pov_combined['Total %'] = '100.0%'
    pov_combined.index.name = 'Province, UR, Sex'
    pov_combined.columns.name = 'Poverty Level'

    # Plotting setup
    plot_data = poverty_percent.reset_index().melt(
        id_vars=['province', 'ur', 'sex'],
        var_name='Poverty Level',
        value_name='Weighted Percentage'
    )

    plt.figure(figsize=(14, 5))
    bar_plot = sns.barplot(
        data=plot_data,
        x='province',
        y='Weighted Percentage',
        hue='Poverty Level',
        ci=None
    )

    # Manually add percentage labels on each bar (since bar_label is unavailable)
    for bars in bar_plot.containers:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                bar_plot.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_y() + height / 2,
                    f'{height:.1f}%',
                    ha='center',
                    va='center',
                    color='white',
                    fontsize=9,
                    weight='bold'
                )

    plt.title('Weighted Poverty Levels by Province')
    plt.ylabel('Weighted Percentage')
    plt.xlabel('Province')
    plt.xticks(rotation=45)
    plt.legend(title='Poverty Level', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    # Save the figure into a BytesIO buffer
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)

    # Encode the image to base64
    img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')

    # Close plot
    plt.close()

    return img_base64

    # return pov_combined
def poverty_level_by_sex_tbl(merged_df):
    # Weighted or unweighted counts
    if 'weight' in merged_df.columns:
        grouped = merged_df.groupby(['province', 'ur', 'sex', 'poverty_level'])['weight'].sum()
    else:
        grouped = merged_df.groupby(['province', 'ur', 'sex', 'poverty_level']).size()

    poverty_counts = grouped.unstack(fill_value=0)
    row_totals = poverty_counts.sum(axis=1)
    poverty_percent = poverty_counts.divide(row_totals, axis=0) * 100

    # Combine counts and percentages into one cell
    pov_combined = poverty_counts.copy()
    for col in poverty_counts.columns:
        pov_combined[col] = poverty_counts[col].astype(int).map(lambda x: f"{x:,.0f}") + \
                        ' (' + poverty_percent[col].map(lambda x: f"{x:.1f}%") + ')'

    pov_combined['Total'] = row_totals.map(lambda x: f"{x:,.0f}")
    pov_combined['Total %'] = '100.0%'
    pov_combined.index.name = 'Province, UR, Sex'
    pov_combined.columns.name = 'Poverty Level'
    tabel = pov_combined
    tabel.columns = [''.join(col).strip() for col in tabel.columns.values]
    tabel.reset_index(inplace=True)

    # tabel = tabel.
    return tabel

   