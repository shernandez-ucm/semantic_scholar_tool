# Project: Transformers for Renewable Energy Forecasting                                                                                                                                    
                                                                                                                                                                                              
  **Topic:** Multi-site forecasting with intermittent time series                                                                                                                             
  **Goal:** Develop transformer-based models for forecasting renewable energy generation across multiple geographic sites with irregular, intermittent patterns                               
                                                                                                                                                                                              
  ---                                                                                                                                                                                         
                                                                                                                                                                                              
  ## Phase 1: Discovery                                                                                                                                                                       
  - **Research Specification** → Define forecasting horizon, sites, variables, evaluation metrics


  - **Literature Review** → Survey transformer applications in energy forecasting, missing data handling, multi-site approaches                                                               
  
### Key Architecture Types                                                                                                                                                                  
                  
  #### 1. Hierarchical Temporal Transformer (HTT)                                                                                                                                             
  - **Structure:** Global (seasonal) + Local (residual) branches merge via skip connections
  - **For intermittents:** Combine with GARCH or zero-inflated components                                                                                                                     
  - **Reference:** Zhou et al. (2022) - "Transformer with hierarchical temporal modeling"                                                                                                     
                                                                                                                                                                                              
  #### 2. Dual-Head Sparse Transformer (DHST)                                                                                                                                                 
  - **Structure:** One head attends globally (global), one head attends locally (local window)                                                                                                
  - **For intermittents:** Global head uses full attention, local head uses sparse window attention                                                                                           
  - **Innovation:** Separates trend/seasonality from local dynamics                                                                                                                           
                                                                                                                                                                                              
  #### 3. Mixture-of-Experts (MoE) with Temporal Routing                                                                                                                                      
  - **Structure:** Global experts learn long-range patterns, local experts learn short-term spikes                                                                                            
  - **For intermittents:** Route sparse arrivals to specialized local experts                                                                                                                 
  - **Reference:** Shazeer et al. extended for temporal data                                                                                                                                  
                                                                                                                                                                                              
  #### 4. Local-Global Factorized Attention (LGFA)                                                                                                                                            
  - **Structure:** Attention matrix factorized: Q @ K_local and Q @ K_global                                                                                                                  
  - **For intermittents:** Global captures demand patterns, local captures zero-events                                                                                                        
  - **Computational:** O(n) instead of O(n²) for large lookback windows                                                                                                                       
                                                                                                                                                                                              
  ### Intermittent-Specific Modifications                                                                                                                                                     
                                                                                                                                                                                              
  #### Zero-Inflated Attention Masking                                                                                                                                                        
  - **Apply:** Zero-out attention scores for zero observations in local windows
  - **Why:** Prevents noise from zeros from contaminating trend learning                                                                                                                      
  - **How:** Use learned attention masks for zero-valued periods                                                                                                                              
                                                                                                                                                                                              
  #### Event-Triggered Sparse Attention                                                                                                                                                       
  - **Apply:** Only attend from previous non-zero events for local modeling                                                                                                                   
  - **Global:** Continue attending normally                                                                                                                                                   
  - **Why:** Irrelevant for continuous processes                                                                                                                                              
                                                                                                                                                                                              
  #### Hybrid GARCH-Transformer                                                                                                                                                               
  - **Apply:** GARCH component models zero-inflation variance, transformer models mean                                                                                                        
  - **Structure:** Global handles variance dynamics, local handles mean dynamics                                                                                                              
  - **Reference:** Hyndman et al. on intermittent demand models                                                                                                                               
                                                                                                                                                                                              
  ### Implementation Guidelines                                                                                                                                                               
                                                                                                                                                                                              
  1. **Global Path:** Full sequence attention (or efficient variants like Linear/Performer)                                                                                                   
     - Purpose: Capture seasonality, long-term trends
     - Window: Full lookback or very large window                                                                                                                                             
     - Attention mechanism: Full attention or low-rank approximation                                                                                                                          
                                                                                                                                                                                              
  2. **Local Path:** Sparse window attention                                                                                                                                                  
     - Purpose: Capture immediate local dynamics                                                                                                                                              
     - Window: Small window (e.g., 7-30 days)                                                                                                                                                 
     - Attention mechanism: Sparse attention, grouped-query, or sliding window                                                                                                                
                                                                                                                                                                                              
  3. **Combination Methods:**                                                                                                                                                                 
     - Skip connections: Global output + Local output -> Merge layer                                                                                                                          
     - Gating: Learn when to rely on global vs local via learned gates                                                                                                                        
     - Ensemble: Both branches make predictions, ensemble average                                                                                                                             
                                                                                                                                                                                              
  4. **For Intermittent Series:**                                                                                                                                                             
     - Add zero-event handling (masking, special embedding)                                                                                                                                   
     - Consider Poisson-like count models for local dynamics                                                                                                                                  
     - Global path can use time-aware embeddings for seasonality                                                                                                                              
                                                                                                                                                                                              
  ### Recommended Configuration                                                                                                                                                               
                                                                                                                                                                                              
  - **Global Window:** Full sequence (or 1-2 years for seasonal)                                                                                                                              
  - **Local Window:** 7-30 days (or one season)
  - **Attention Heads:** Separate head counts (e.g., 8 global, 4 local)                                                                                                                       
  - **Merge Strategy:** Linear combination + learned weights                                                                                                                                  
  - **Zero-Handling:** Per-token zero embeddings + attention masking                                                                                                                          
                                                                                                                                                                                              
  ### Key References                                                                                                                                                                                                                                                                                                                                                
  - **TSMixer:** Global-local mixing for time-series                                                                                                                                          
  - **PatchTST:** Patch-based global modeling
  - **iTransformer:** Inverted structure for multi-variate                                                                                                                                    
  - **Zero-Inflated Transformer:** For sparse count data                                                                                                                                      
  - **Hybrid models:** LSTM-Transformer, ARIMA-Transformer hybrids                                                                                                                            
                                            

  - **Data Assessment** → Identify datasets: wind/solar generation, weather data, grid consumption                                                                                            
                                                                                                                                                                                       
  ## Phase 2: Strategy                                                                                                                                                                        
  - **Identification Strategy** → Multi-task learning vs. hierarchical models vs. federated approaches                                                                                        
  - **Data Strategy** → Handling intermittency (missing periods, low generation) and site correlations                                                                                        
  - **Robustness Plan** → Sensitivity to site correlation structures, temporal patterns                                                                                                       
                                                                                                                                                                                              
  ## Phase 3: Execution                                                                                                                                                                       
  - **Data Pipeline** → Cleaning, feature engineering (time, weather, site covariates)                                                                                                        
  - **Model Development** → Transformer architectures (ViT, T5-style, Informer, PatchTST variants)                                                                                            
  - **Evaluation** → Multi-site metrics: MAE, RMSE, directional accuracy, cross-site transfer                                                                                                 
                                                                                                                                                                                              
  ## Phase 4: Peer Review                                                                                                                                                                     
  - **Quality Review** → Internal consistency, methodological soundness                                                                                                                       
  - **External Review** → Simulated domain and methods referees                                                                                                                               
                                                                                                                                                                                              
  ## Phase 5: Submission                                                                                                                                                                      
  - **Target Journals** → Energy Economics, Journal of Cleaner Production, Energy, Applied Energy                                                                                             
  - **Replication Package** → All data, code, results                                                                                                                                         
  - **Final Verification** → Quality gate >= 95 before submission        