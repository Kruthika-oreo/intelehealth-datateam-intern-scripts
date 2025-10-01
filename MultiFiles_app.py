import streamlit as st
import pandas as pd
import re
import matplotlib.pyplot as plt


st.set_page_config(layout="wide")
st.title("NAS Data Analysis Dashboard")
# ------------------------
# 1. Upload CSV/Excel (single or multiple)
# ------------------------
uploaded_files = st.file_uploader("Upload NAS CSV/XLSX file(s)", type=["csv","xlsx"], accept_multiple_files=True)

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
    st.subheader("Top 20 Diagnoses")
    diagnosis_counts = df['Primary & Provisional'].value_counts()
    fig, ax = plt.subplots(figsize=(10,6))
    diagnosis_counts.head(20).plot(kind='barh', color='red', ax=ax)
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Diagnosis")
    ax.set_title("Top 20 Diagnoses in NAS")
    plt.gca().invert_yaxis()
    st.pyplot(fig)
    
    # ------------------------
    # 4. Single vs Multiple Diagnoses Top 20
    # ------------------------
    st.subheader("Top 20 Single Diagnoses")
    fig, ax = plt.subplots(figsize=(10,6))
    single_diag_df['Primary & Provisional'].value_counts().head(20).plot(kind='barh', color='red', ax=ax)
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Single Diagnosis")
    ax.set_title("Top 20 Single Diagnoses")
    plt.gca().invert_yaxis()
    st.pyplot(fig)

    st.subheader("Top 20 Multiple Diagnoses")
    fig, ax = plt.subplots(figsize=(10,6))
    multiple_diag_df['Primary & Provisional'].value_counts().head(20).plot(kind='barh', color='red', ax=ax)
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Multiple Diagnosis")
    ax.set_title("Top 20 Multiple Diagnoses")
    plt.gca().invert_yaxis()
    st.pyplot(fig)

    # ------------------------
    # 5. Images Stats
    # ------------------------
    df['Has_image'] = df['Images'].str.contains('http', na=False)
    image_counts = df['Has_image'].value_counts()
    total_rows = len(df)
    
    st.subheader("Patients with vs without Images")
    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar(image_counts.index.map({True: 'Has Image', False: 'No Image'}), image_counts.values, color='red')
    ax.set_xlabel("Image Availability")
    ax.set_ylabel("Number of Patients")
    ax.set_title(f"Patients with vs without Images (Total Rows={total_rows})")
    for i, v in enumerate(image_counts.values):
        ax.text(i, v+1, str(v), ha='center')
    st.pyplot(fig)

    # ------------------------
    # 6. Single/Multiple for sub-samples (with and without images)
    # ------------------------
    st.subheader("Top 20 Single/Multiple Diagnoses - Patients with Images")
    df_with_images = df[df['Has_image']==True]
    fig, ax = plt.subplots(figsize=(10,6))
    df_with_images[df_with_images['Diagnosis_type']=='Single']['Primary & Provisional'].value_counts().head(20).plot(kind='barh', color='red', ax=ax)
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Single Diagnosis")
    ax.set_title("Top 20 Single Diagnoses - Patients with Images")
    plt.gca().invert_yaxis()
    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(10,6))
    df_with_images[df_with_images['Diagnosis_type']=='Multiple']['Primary & Provisional'].value_counts().head(20).plot(kind='barh', color='red', ax=ax)
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Multiple Diagnosis")
    ax.set_title("Top 20 Multiple Diagnoses - Patients with Images")
    plt.gca().invert_yaxis()
    st.pyplot(fig)

    st.subheader("Top 20 Single/Multiple Diagnoses - Patients without Images")
    df_no_images = df[df['Has_image']==False]
    fig, ax = plt.subplots(figsize=(10,6))
    df_no_images[df_no_images['Diagnosis_type']=='Single']['Primary & Provisional'].value_counts().head(20).plot(kind='barh', color='red', ax=ax)
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Single Diagnosis")
    ax.set_title("Top 20 Single Diagnoses - Patients without Images")
    plt.gca().invert_yaxis()
    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(10,6))
    df_no_images[df_no_images['Diagnosis_type']=='Multiple']['Primary & Provisional'].value_counts().head(20).plot(kind='barh', color='red', ax=ax)
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Multiple Diagnosis")
    ax.set_title("Top 20 Multiple Diagnoses - Patients without Images")
    plt.gca().invert_yaxis()
    st.pyplot(fig)

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
    med_freq.head(20).plot(kind='barh', color='red', ax=ax)
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Medicine")
    ax.set_title("Top 20 Prescribed Medicines")
    plt.gca().invert_yaxis()
    st.pyplot(fig)

    st.subheader("All Prescribed Medicines")
    fig, ax = plt.subplots(figsize=(12,10))
    med_freq.plot(kind='barh', color='red', ax=ax)
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Medicine")
    ax.set_title("All Prescribed Medicines Frequency")
    plt.gca().invert_yaxis()
    st.pyplot(fig)
    
    # ------------------------
    # 8. Referrals
    # ------------------------
    phc_count = df['Referral_advice'].str.contains('PHC', na=False).sum()
    dh_count = df['Referral_advice'].str.contains('DH', na=False).sum()
    sdh_count = df['Referral_advice'].str.contains('SDH', na=False).sum()
    
    referral_counts = pd.DataFrame({
        'Facility': ['PHC','DH','SDH'],
        'Count': [phc_count, dh_count, sdh_count]
    })
    
    st.subheader("Referrals Suggested by Facility")
    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar(referral_counts['Facility'], referral_counts['Count'], color='red')
    ax.set_xlabel("Facility")
    ax.set_ylabel("Number of Referrals")
    ax.set_title("Referrals Suggested by Facility")
    for i, v in enumerate(referral_counts['Count']):
        ax.text(i, v+1, str(v), ha='center')
    st.pyplot(fig)

    # ------------------------
    # 9. Visit Dates Range & Monthly Stratifications
    # ------------------------
    df['Visit_started_date'] = pd.to_datetime(df['Visit_started_date'], errors='coerce')
    dates = df['Visit_started_date'].dropna()
    start_date = dates.min()
    end_date = dates.max()
    total_days = (end_date - start_date).days + 1
    total_visits = len(dates)
    
    st.write(f"Visit date range: {start_date.date()} to {end_date.date()} ({total_days} days)")
    st.write(f"Total visits: {total_visits}")
    
    df['Month'] = df['Visit_started_date'].dt.to_period('M')
    
    st.subheader("Gender Stratification by Month")
    gender_month = df.groupby(['Month','Gender']).size().unstack(fill_value=0)
    st.dataframe(gender_month)
    gender_month.plot(kind='bar', figsize=(10,6), color=['red','lightblue'])
    plt.xlabel("Month")
    plt.ylabel("Number of Patients")
    plt.title("Gender Stratification of Patients by Month")
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(plt.gcf())
    
    st.subheader("Age Stratification by Month")
    bins = [0,12,18,35,50,65,100]
    labels = ['0-12','13-18','19-35','36-50','51-65','66+']
    df['Age_Group'] = pd.cut(df['Age'], bins=bins, labels=labels, right=True)
    age_month = df.groupby(['Month','Age_Group']).size().unstack(fill_value=0)
    st.dataframe(age_month)
    age_month.plot(kind='bar', stacked=True, figsize=(12,6), colormap='Reds')
    plt.xlabel("Month")
    plt.ylabel("Number of Patients")
    plt.title("Age Group Stratification by Month")
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(plt.gcf())
