# SPLHRNMTF
MicroRNAs (miRNAs) have been demonstrated to be closely related to human diseases. Studying the potential associations between miRNAs and diseases contributes to our understanding of disease pathogenic mechanisms. As traditional biological experiments are costly and time-consuming, computational models can be considered as effective complementary tools. In this study, we propose a novel model of robust orthogonal non-negative matrix tri-factorization (NMTF) with self-paced learning and dual hypergraph regularization, named SPLHRNMTF, to predict miRNA-disease associations. More specifically, SPLHRNMTF first uses a non-linear fusion method to obtain miRNA and disease comprehensive similarity. Subsequently, the improved miRNA-disease association matrix is reformulated based on weighted k-nearest neighbor profiles to correct false-negative associations. In addition, we utilize L2,1 norm to replace Frobenius norm to calculate residual error, alleviating the impact of noise and outliers on prediction performance. Then, we integrate self-paced learning into NMTF to alleviate the model from falling into bad local optimal solutions by gradually including samples from easy to complex. Finally, hypergraph regularization is introduced to capture high-order complex relations from hypergraphs related to miRNAs and diseases. In 5-fold cross-validation five times experiments, SPLHRNMTF obtains higher average AUC values than other baseline models. Moreover, the case studies on breast neoplasms and lung neoplasms further demonstrate the accuracy of SPLHRNMTF. Meanwhile, the potential associations discovered are of biological significance.

# The workflow of SPLHRNMTF model
![The workflow of SPLHRNMTF model](https://github.com/Ouyang-Dong/HGCLAMIR/blob/master/workflow.jpg)
# Environment Requirement
The code has been tested running under Python 3.13. The required packages are as follows:
- numpy == 1.26.0
- pandas == 1.5.3
