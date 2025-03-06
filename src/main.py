import pandas as pd
from extract_text import extract_text_from_pdf, clean_text
from topic_modelling import detect_topics
from llm_annotate import process_sections, aggregate_results

pdf_path = "../data/sample_ethnography.pdf"  

full_text = extract_text_from_pdf(pdf_path)

cleaned_text = clean_text(full_text)

sections = detect_topics(cleaned_text)

section_results = process_sections(sections)

if section_results:
    final_results = aggregate_results(section_results)
    df = pd.DataFrame([final_results])
    df.to_csv("../results/final_coded_ethnography.csv", index=False)
else:
    print("No results generated.")