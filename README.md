<!-- PROJECT LOGO -->
<br />
<div align="center">




  <h3 align="center">Predicting Complications and Mortality In Myocardial Infarction Patients Using A Graph Neural Network Model </h3>
  <img width="2390" height="682" alt="Main model" src="https://github.com/user-attachments/assets/e8f04f62-5742-4321-bbef-052ad031a008" />

  <p align="justify">
       Myocardial infarction results from sudden coronary artery blockage, leading to myocardial necrosis and severe complications. Early prediction is critical to mitigate these life-threatening outcomes and improve patient survival, but it remains challenging due to data heterogeneity and limited temporal feature utilization. To address this crucial gap, we proposed a GraphTransformer-based model with three key innovations: First, a density-adaptive optimized K-Nearest Neighbor graph to model patient similarities dynamically. Second, a short-term Convolutional Neural Network for fine-grained temporal feature extraction with dynamic gating. Third, a long-term Gated Recurrent Unit with Mamba-inspired dynamic gating to capture robust long-term dependencies. Experiment on the Myocardial Infarction Complications dataset, our model achieved an average AUC of 0.7236, with a standout 0.8777 for mortality prediction. Moreover, SHapley Additive exPlanations analysis revealed critical features like serum sodium levels and temporal dynamics, and enhanced clinical interpretability. Leveraging automated feature extraction from admission data, 72-hour temporal sequences, and derived temporal features with missing values, our approach empowers a robust DL model for multi-outcome prediction. Our method enables timely interventions, significantly reducing mortality and rehospitalization rates while equipping clinicians with a vital tool to boost patient survival, optimize critical care resources, and support tailored treatment strategies for MI patients' dynamic needs. <br />
    <br />
    The source code of HyCoSNet will be made public after the paper is accepted.
    <br />
  </p>
   <h3 align="lift">Dataset </h3>
  <p align="lift">
    Myocardial infarction complications (https://archive.ics.uci.edu/dataset/579/myocardial+infarction+complications)
<br />
    Golovenkin, S., Shulman, V., Rossiev, D., Shesternya, P., Nikulina, S., Orlova, Y., & Voino-Yasenetsky, V. (2020). Myocardial infarction complications [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C53P5M.
   </p>
</div>
