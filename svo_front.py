import streamlit as st
from pipeline import pipeline
import os


with st.sidebar:
    st.title('БИОКАД')
col1, col2 = st.columns(2)
with col1:
    mol1 = st.text_input("vl 1", "")
    mol2 = st.text_input("vh 1", "")
with col2:
    tar1 = st.text_input("vl 2", "")
    tar2 = st.text_input("vh 2", "")


def make():
    if mol1 != "" and mol2 != "" and tar1 != "" and tar2 != "":
        sc_start, start_losses, answer, end_losses = pipeline(mol1, mol2, tar1, tar2)
        print(sc_start, start_losses, answer, end_losses)
        sc_end, ans = answer
        st.text(ans.vh.seq)
        st.text(ans.vl.seq)
        st.text('start: ' + str(sc_start) + f'    Losses: {start_losses[0]} v_gene_l    {start_losses[1]} v_gene_h   {start_losses[2]} RMSD')
        st.text('new: ' + str(sc_end) + f'    Losses: {end_losses[0]} v_gene_l    {end_losses[1]} v_gene_h   {end_losses[2]} RMSD')
        st.download_button('Скачать ответ', bytes('>H\n' + ans.vh.seq + '\n>L\n' + ans.vl.seq, encoding='utf-8'), 'answer.fasta')

if st.button("Запуск"):
    make()
