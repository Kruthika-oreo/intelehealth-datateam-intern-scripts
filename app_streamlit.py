import streamlit as st
import pandas as pd
import re
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("NAS Data Analysis Dashboard")

# ------------------------
# 1. Upload CSV/Excel (single or multiple)
# ------------------------
uploaded_files = st.file_uploader(
    "Upload NAS CSV/XLSX file(s)",
    type=["csv", "xlsx"],
    accept_multiple_files=True,
    key="nas_file_uploader"
)

if uploaded_files:
    df_list = []

    for file in uploaded_files:
        if file.name.endswith('.csv'):
            temp_df = pd.read_csv(file)
        else:
            temp_df = pd.read_excel(file)
        st.write(f"{file.name} uploaded: {len(temp_df)} rows")
        df_list.append(temp_df)

    # Merge all uploaded files
    df = pd.concat(df_list, ignore_index=True)
    st.success(f"All files merged! Total rows: {len(df)}")

    # ------------------------
    # 2. Diagnosis Classification
    # ------------------------
    def classify_diagnosis(diagnosis):
        if pd.isna(diagnosis):
            return 'Unknown'
        diag = diagnosis.strip()
        if ',' in diag:
            after_comma = diag.split(',', 1)[1].strip()
            if len(after_comma) > 5:
                return 'Multiple'
            else:
                return 'Single'
        if re.search(r'[a-z][A-Z]', diag):
            return 'Multiple'
        return 'Single'

    df['Diagnosis_type'] = df['Primary & Provisional'].apply(classify_diagnosis)

    # Separate single and multiple
    single_diag_df = df[df['Diagnosis_type'] == 'Single']
    multiple_diag_df = df[df['Diagnosis_type'] == 'Multiple']

    # ------------------------
    # 3. Top 20 Diagnoses
    # ------------------------
    def add_labels(ax):
        """Add data labels to a horizontal or vertical bar chart"""
        for container in ax.containers:
            ax.bar_label(container, fmt='%d', label_type='edge', padding=3)

    st.subheader("Top 20 Diagnoses")
    diagnosis_counts = df['Primary & Provisional'].value_counts()
    fig, ax = plt.subplots(figsize=(10,6))
    bars = diagnosis_counts.head(20).plot(kind='barh', color='red', ax=ax)
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Diagnosis")
    ax.set_title("Top 20 Diagnoses in NAS")
    plt.gca().invert_yaxis()
    add_labels(ax)
    st.pyplot(fig)

    # ------------------------
    # 4. Single vs Multiple Diagnoses Top 20
    # ------------------------
    st.subheader("Top 20 Single Diagnoses")
    fig, ax = plt.subplots(figsize=(10,6))
    bars = single_diag_df['Primary & Provisional'].value_counts().head(20).plot(kind='barh', color='red', ax=ax)
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Single Diagnosis")
    ax.set_title("Top 20 Single Diagnoses")
    plt.gca().invert_yaxis()
    add_labels(ax)
    st.pyplot(fig)

    st.subheader("Top 20 Multiple Diagnoses")
    fig, ax = plt.subplots(figsize=(10,6))
    bars = multiple_diag_df['Primary & Provisional'].value_counts().head(20).plot(kind='barh', color='red', ax=ax)
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Multiple Diagnosis")
    ax.set_title("Top 20 Multiple Diagnoses")
    plt.gca().invert_yaxis()
    add_labels(ax)
    st.pyplot(fig)

    # ------------------------
    # 5. Images Stats
    # ------------------------
    df['Has_image'] = df['Images'].str.contains('http', na=False)
    image_counts = df['Has_image'].value_counts()
    total_rows = len(df)

    st.subheader("Patients with vs without Images")
    fig, ax = plt.subplots(figsize=(6,4))
    bars = ax.bar(image_counts.index.map({True: 'Has Image', False: 'No Image'}), image_counts.values, color='red')
    ax.set_xlabel("Image Availability")
    ax.set_ylabel("Number of Patients")
    ax.set_title(f"Patients with vs without Images (Total Rows={total_rows})")
    add_labels(ax)
    st.pyplot(fig)

    # ------------------------
    # 6. Single/Multiple for sub-samples (with and without images)
    # ------------------------
    st.subheader("Top 20 Single/Multiple Diagnoses - Patients with Images")
    df_with_images = df[df['Has_image']==True]
    fig, ax = plt.subplots(figsize=(10,6))
    bars = df_with_images[df_with_images['Diagnosis_type']=='Single']['Primary & Provisional'].value_counts().head(20).plot(kind='barh', color='red', ax=ax)
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Single Diagnosis")
    ax.set_title("Top 20 Single Diagnoses - Patients with Images")
    plt.gca().invert_yaxis()
    add_labels(ax)
    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(10,6))
    bars = df_with_images[df_with_images['Diagnosis_type']=='Multiple']['Primary & Provisional'].value_counts().head(20).plot(kind='barh', color='red', ax=ax)
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Multiple Diagnosis")
    ax.set_title("Top 20 Multiple Diagnoses - Patients with Images")
    plt.gca().invert_yaxis()
    add_labels(ax)
    st.pyplot(fig)

    st.subheader("Top 20 Single/Multiple Diagnoses - Patients without Images")
    df_no_images = df[df['Has_image']==False]
    fig, ax = plt.subplots(figsize=(10,6))
    bars = df_no_images[df_no_images['Diagnosis_type']=='Single']['Primary & Provisional'].value_counts().head(20).plot(kind='barh', color='red', ax=ax)
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Single Diagnosis")
    ax.set_title("Top 20 Single Diagnoses - Patients without Images")
    plt.gca().invert_yaxis()
    add_labels(ax)
    st.pyplot(fig)

    multiple_no_images = df_no_images[df_no_images['Diagnosis_type']=='Multiple']['Primary & Provisional'].value_counts().head(20)
    if not multiple_no_images.empty:
        fig, ax = plt.subplots(figsize=(10,6))
        bars = multiple_no_images.plot(kind='barh', color='red', ax=ax)
        ax.set_xlabel("Frequency")
        ax.set_ylabel("Multiple Diagnosis")
        ax.set_title("Top 20 Multiple Diagnoses - Patients without Images")
        plt.gca().invert_yaxis()
        add_labels(ax)
        st.pyplot(fig)
    else:
        st.write("No patients without images have multiple diagnoses in this dataset.")

    # ------------------------
    # 7. Medications
    # ------------------------
    def split_medicines(row):
        if pd.isna(row):
            return []
        parts = re.split(r',(?![^()]*\))', row)
        return [p.strip().rstrip('.') for p in parts if p.strip()]

    df['Medicines_split'] = df['Medicines'].apply(split_medicines)
    all_meds = df['Medicines_split'].explode().reset_index(drop=True)
    med_freq = all_meds.value_counts()

    st.subheader("Top 20 Prescribed Medicines")
    fig, ax = plt.subplots(figsize=(10,6))
    bars = med_freq.head(20).plot(kind='barh', color='red', ax=ax)
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Medicine")
    ax.set_title("Top 20 Prescribed Medicines")
    plt.gca().invert_yaxis()
    add_labels(ax)
    st.pyplot(fig)

    st.subheader("All Prescribed Medicines")
    fig, ax = plt.subplots(figsize=(12,10))
    bars = med_freq.plot(kind='barh', color='red', ax=ax)
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Medicine")
    ax.set_title("All Prescribed Medicines Frequency")
    plt.gca().invert_yaxis()
    add_labels(ax)
    st.pyplot(fig)

    # ------------------------
    # 8. Referrals
    # ------------------------
    phc_count = df['Referral_advice'].str.contains('PHC', na=False).sum()
    dh_count = df['Referral_advice'].str.contains('DH', na=False).sum()
    sdh_count = df['Referral_advice'].str.contains('SDH', na=False).sum()

    referral_counts = pd.DataFrame({
        'Facility': ['PHC', 'DH', 'SDH'],
        'Count': [phc_count, dh_count, sdh_count]
    })

    st.subheader("Referrals Suggested by Facility")
    fig, ax = plt.subplots(figsize=(6,4))
    bars = ax.bar(referral_counts['Facility'], referral_counts['Count'], color='red')
    ax.set_xlabel("Facility")
    ax.set_ylabel("Number of Referrals")
    ax.set_title("Referrals Suggested by Facility")
    add_labels(ax)
    st.pyplot(fig)

    # ------------------------
    # Visit Date Range
    # ------------------------
    df['Visit_started_date'] = pd.to_datetime(df['Visit_started_date'], errors='coerce')
    dates = df['Visit_started_date'].dropna()
    start_date = dates.min()
    end_date = dates.max()
    total_days = (end_date - start_date).days + 1
    total_visits = len(dates)

    st.write(f"**Visit date range:** {start_date.date()} to {end_date.date()} ({total_days} days)")
    st.write(f"**Total visits:** {total_visits}")

    # ------------------------
    # Month Extraction
    # ------------------------
    df['Month'] = df['Visit_started_date'].dt.to_period('M')
    df['Month_Label'] = df['Visit_started_date'].dt.strftime('%Y-%m (%b)')

    # ------------------------
    # Gender Stratification
    # ------------------------
    st.subheader("Gender Stratification by Month")
    df['Gender_Label'] = df['Gender'].map({'M': 'Male', 'F': 'Female'}).fillna('Other')
    gender_month = df.groupby(['Month_Label','Gender_Label'], observed=False).size().unstack(fill_value=0)
    st.dataframe(gender_month)

    gender_colors = {'Male': 'blue', 'Female': 'pink', 'Other': 'purple'}
    ax = gender_month.plot(
        kind='bar',
        figsize=(10,6),
        color=[gender_colors.get(col, 'gray') for col in gender_month.columns]
    )
    plt.xlabel("Month")
    plt.ylabel("Number of Patients")
    plt.title("Gender Stratification of Patients by Month")
    plt.xticks(rotation=45)
    plt.legend(title="Gender")
    plt.tight_layout()

    for container in ax.containers:
        ax.bar_label(container)
    st.pyplot(plt.gcf())

    # ------------------------
    # Age Stratification
    # ------------------------
    st.subheader("Age Stratification by Month")
    bins = [0, 12, 18, 59, 200]
    labels = ['Pediatric (0-12)', 'Adolescent (13-18)', 'Adults (19-59)', 'Elderly (60+)']
    df['Age_Group'] = pd.cut(df['Age'], bins=bins, labels=labels, right=True)

    age_month = df.groupby(['Month_Label','Age_Group'], observed=False).size().unstack(fill_value=0)
    st.dataframe(age_month)

    age_colors = {
        'Pediatric (0-12)': 'yellow',
        'Adolescent (13-18)': 'orange',
        'Adults (19-59)': 'teal',
        'Elderly (60+)': 'purple'
    }

    ax = age_month.plot(
        kind='bar',
        stacked=True,
        figsize=(12,6),
        color=[age_colors.get(col, 'gray') for col in age_month.columns]
    )
    plt.xlabel("Month")
    plt.ylabel("Number of Patients")
    plt.title("Age Group Stratification by Month")
    plt.xticks(rotation=45)
    plt.legend(title="Age Group")
    plt.tight_layout()

    for container in ax.containers:
        ax.bar_label(container)
    st.pyplot(plt.gcf())
