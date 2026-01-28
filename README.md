<br />
<div align="center">


  <h3 align="center">Predicting Complications and Mortality In Myocardial Infarction Patients Using A Graph Neural Network Model </h3>
  <img width="468" height="202" alt="image" src="https://github.com/user-attachments/assets/b00676c1-584f-43f3-a926-9eb23a2287b9" />


  <p align="justify">
       MMyocardial infarction (MI) is often complicated by heterogeneous life-threatening conditions, which require outcome-specific risk stratification. Current models typically predict a single composite endpoint and fail to fully exploit inter-patient similarities and temporal dynamics in electronic health records. To address this crucial gap,  we present the first graph neural network framework that simultaneously predicts 12 distinct post-MI complications and in-hospital mortality. The model integrates three targeted innovations: first, a density-adaptive K-nearest neighbor graph to capture clinically meaningful patient similarities; second, dual-branch short- and long-term temporal encoders with dynamic gating; and third, cross-modal attention for interactive fusion of multi-scale temporal features. Experiment on a 1,700-patient MI complications dataset, our model achieved an average AUC of 0.7330, with a standout 0.8828 for mortality prediction. SHAP analysis and built-in attention weights identified age, serum sodium, and dynamic laboratory trends as top predictors, aligning with clinical knowledge. This interpretable approach offers potential for early, individualized risk assessment in acute cardiac care.   <br />
    <br />
    The source code of our GNN model has been published as the paper has been accepted.
    <br />
  </p>
   <h3 align="lift">Dataset </h3>
  <p align="lift">
    Myocardial infarction complications (https://archive.ics.uci.edu/dataset/579/myocardial+infarction+complications)
<br />
    Golovenkin, S., Shulman, V., Rossiev, D., Shesternya, P., Nikulina, S., Orlova, Y., & Voino-Yasenetsky, V. (2020). Myocardial infarction complications [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C53P5M.
   </p>
</div>
