# CudaParticleMotionLaw

GPU accelerated particle motion law

Based on "How life emerges from a simple particle motion law: Introducing the Primordial Particle System"    

# Controls  

|Event|Action|  
|---|---|  
|**Mouse Middle**|Panning|  
|**Mouse Wheel**|Zooming In/Out|  
|**Esc**|Close window|  
|**P**|Pause/Unpause|  
|**Arrow Up/Down**|Change simulation bounds|  
|**Mouse Left button**|Spawn group of particles(hard coded 690)|  
|**M**|	Switch between CPU/GPU|  

# Settings

Parameters of simulation can be changed via configuration files in resources folder.

# Peformance

O(n^2) Collision detection optimization can easily increase performance of both methods.   

|Processor Type \ Particle Number|2000|5000|10000|  
|---|---|---|---|
|**CPU**|8ms|45ms|175ms|  
|**GPU**|3ms|7ms|13ms|  

# Visualisation

Configuration 1   

![Animation](https://github.com/SlawoStr/CudaParticleMotionLaw/blob/master/img/Animation.gif)

Configuration 2

![Animation](https://github.com/SlawoStr/CudaParticleMotionLaw/blob/master/img/Animation2.gif)
