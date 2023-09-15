
for j in {0..10}; 
  do i=$((10*j)); 
  echo ${i}; 
  data_name="${i}_zf_${i}_b"; 
  f="Data/BindZFpredictor/${data_name}"; 
  mkdir -p $f; 
  mkdir -p ${f}/predictions; 
  
  python BindZF_predictor/code/main_loo_bindzfpredictor.py 
    -b_n ${data_name} 
    -b_d Data/BindZFpredictor/ 
    -m_d BindZF_predictor/code 
    -r 1 
    -p_add ${f}/predictions;
done