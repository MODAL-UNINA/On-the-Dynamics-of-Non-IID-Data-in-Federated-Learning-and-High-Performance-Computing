# On the Dynamics of Non-IID Data in Federated Learning and High Performance Computing

This is the implemetation code for the paper (link): Daniela Annunziata, Marzia Canzaniello, Diletta Chiaro, Stefano Izzo, Martina Savoia, Francesco Piccialli, On the Dynamics of Non-IID Data in Federated Learning and High-Performance Computing.

**Abstract**

This paper investigates the symbiosis of Federated Learning (FL) and High-Performance Computing (HPC) architectures, unraveling challenges introduced by the intricate interplay of heterogeneity and non-Independently and Identically Distributed (non-IID) data. By leveraging the Flower framework, our research delves into the nuanced implications of FL in diverse HPC environments. We provide a comprehensive exploration of the heterogeneity within contemporary HPC architectures, spanning node organizations, memory hierarchies, and specialized accelerators, emphasizing adaptability to this complexity. Methodologically, we simulate a FL scenario within our research laboratory, leveraging Flower to orchestrate collaborative model training across heterogeneous nodes. The experiments involve variations in the Dirichlet beta parameter, offering insights into the effects of non-IID data. Results encompass communication efficiency, energy efficiency, and global model accuracy, providing a holistic understanding of the performances across diverse HPC infrastructures. 
This research contributes to the ongoing discourse on efficient and scalable algorithms, providing insights for collaborative learning in the era of diverse HPC architectures.

## Acknowledgments
- PNRR project FAIR -  Future AI Research (PE00000013), Spoke 3, under the NRRP MUR program funded by the NextGenerationEU.
- PNRR Centro Nazionale HPC, Big Data e Quantum Computing, (CN\_00000013)(CUP: E63C22000980007), under the NRRP MUR program funded by the NextGenerationEU.
- G.A.N.D.A.L.F. - Gan Approaches for Non-iiD Aiding Learning in Federations, CUP: E53D23008290006, PNRR - Missione 4 “Istruzione e Ricerca” - Componente C2 Investimento 1.1 “Fondo per il Programma Nazionale di Ricerca e Progetti  di Rilevante Interesse Nazionale (PRIN)”.
- PNRR Centro Nazionale HPC, Big Data e Quantum Computing, (CN\_00000013)(CUP: E63C22000980007), under the NRRP MUR program funded by the NextGenerationEU.

### Usage: 
To run the Federated Learning process, execute the following command, varying the worker_id number, the device and the server IP.
```python
python client.py --worker_id 1 --device 0 --server_IP "localhost:8080"
```
