# CFlowSim
previously *Physics in the Browser*

[![Deployment CI](https://github.com/techlauncher-mlai-edge-physics/techlauncher-mlai-edge-physics.github.io/actions/workflows/main.yml/badge.svg)](https://github.com/techlauncher-mlai-edge-physics/techlauncher-mlai-edge-physics.github.io/actions/workflows/main.yml)

**Deployed at [GitHub Pages](https://techlauncher-mlai-edge-physics.github.io)**

This project is a web application that allows users to simulate the fluid dynamics of a 2D fluid system enhanced by machine learning. The application is built using [React](https://reactjs.org/) and [Three.js](https://threejs.org/). 
The machine learning model is built using [PyTorch](https://pytorch.org/) and converted to [ONNX](https://onnx.ai/) format for inference in the browser using [onnxruntime-web](https://npmjs.com/package/onnxruntime-web).
We have also port the model to [TensorFlow](https://www.tensorflow.org/) so we could use [TensorFlow.js](https://www.tensorflow.org/js) to run the model in the browser,
especially we could benefit from the **WebGL** backend and **WebGPU** backend to speed up the inference.

## About the project

Welcome to our interactive online neural physics emulation engine! What you're seeing is a simulation of computational fluid dynamics, powered by cutting-edge neural network technology.

Computational fluid dynamics is an important area of study that helps us understand how fluids behave in various scenarios, from the flow of air around an airplane to the movement of water in a river. It's a complex field, requiring specialised hardware and software to perform accurate simulations.
However, with the development of neural physics emulation, we can now simulate fluid dynamics using neural networks — a type of machine learning algorithm inspired by the human brain. This approach has the potential to be much faster and more efficient than traditional methods, allowing us to explore complex scenarios and gain new insights into the behaviour of fluids.

### Applications

The ability to simulate fluid dynamics using neural network technology has significant potential for industrial applications.

For example, in the field of aerospace engineering, accurate simulations of fluid dynamics are critical for designing aircraft that are safe, efficient, and effective. By using neural network technology to perform these simulations, engineers can gain a more comprehensive understanding of how air flows around different parts of the aircraft, helping them to optimise its design and performance.

In the automotive industry, neural physics emulation can be used to simulate the flow of air around vehicles, which is important for improving aerodynamics and fuel efficiency. This technology can also be used in the design of cooling systems for engines and other components, ensuring that they operate at the optimal temperature.

In the energy sector, neural physics emulation can be used to simulate the behaviour of fluids in oil and gas wells, helping to optimise the extraction process and improve safety. This technology can also be used in the design of wind turbines and other renewable energy technologies, helping to optimize their efficiency and output.

Overall, the ability to simulate fluid dynamics using neural network technology has broad applications across many industries, helping to improve safety, efficiency, and performance.

### Accessibility

What's really exciting about this online neural physics emulation engine is that it allows anyone with an internet connection to experiment with these simulations themselves. Traditionally, these simulations require specialized hardware and software, and may only be accessible to researchers with access to high-performance computing resources. However, by hosting technology on a web browser, we can make it accessible to anyone, regardless of their technical background.

Our interactive online neural physics emulation engine is based on the research of the Hybrid Prediction Activity, which is part of the Machine Learning and Artificial Intelligence Future Science Platform (MLAI-FSP) in CSIRO's Data61. The MLAI-FSP is dedicated to advancing the field of machine learning and artificial intelligence through interdisciplinary research and collaboration.

## About the team

This project was developed in collaboration with team from ANU TechLauncher, a student industry collaboration program at the Australian National University.

The team is led by students of ANU College of Engineering, Computing and Cybernetics. Students with different levels of expertise in coding, design, content creation, and project management actively collaborate to create this website that presents a user-friendly simulation platform to the public.

## Getting Started

The public deployment of the application can be found at [GitHub Pages](https://techlauncher-mlai-edge-physics.github.io). 
To build and run the application locally, see our [getting started page](docs/gettingstarted.md)

## Contributing

See our [contribution guide](CONTRIBUTION.md) to get started.

## License

Distributed under the BSD-3 Clause License. See `LICENSE` for more information.
