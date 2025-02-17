import * as ort from 'onnxruntime-web/webgpu';
import { type Vector2 } from 'three';
import { type ModelService } from './modelService';

export default class ONNXService implements ModelService {
  session: ort.InferenceSession | null;
  gridSize: [number, number];
  batchSize: number;
  channelSize: number;
  outputChannelSize: number;
  mass: number;
  fpsLimit: number;

  private tensorShape: [number, number, number, number];
  private tensorSize: number;
  private outputSize: number;
  private outputCallback!: (data: Float32Array) => void;
  private matrixArray: Float32Array;
  // 0: partial density
  // 1, 2: partial velocity
  // 3, 4: Force (currently not used)

  private isPaused: boolean;
  private curFrameCountbyLastSecond: number;
  // hold constructor private to prevent direct instantiation
  // ort.InferenceSession.create() is async,
  // so we need to use a static async method to create an instance
  private constructor() {
    this.session = null;
    this.matrixArray = new Float32Array();
    // matrixData and matrixTensor must be sync.
    this.gridSize = [0, 0];
    this.batchSize = 0;
    this.tensorShape = [0, 0, 0, 0];
    this.tensorSize = 0;
    this.outputSize = 0;
    this.isPaused = true;
    this.channelSize = 0;
    this.outputChannelSize = 0;
    this.mass = 0;
    this.fpsLimit = 30;
    this.curFrameCountbyLastSecond = 0;
  }

  // static async method to create an instance
  static async createService(
    modelPath: string,
    gridSize: [number, number] = [64, 64],
    batchSize = 1,
    channelSize = 5,
    outputChannelSize = 3,
    fpsLimit = 15,
    backend = 'wasm',
  ): Promise<ONNXService> {
    console.log('createModelService called');
    const modelServices = new ONNXService();
    await modelServices.init(
      modelPath,
      gridSize,
      batchSize,
      channelSize,
      outputChannelSize,
      backend,
    );
    modelServices.fpsLimit = fpsLimit;
    console.log('createModelService finished');
    return modelServices;
  }

  bindOutput(callback: (data: Float32Array) => void): void {
    this.outputCallback = callback;
  }

  startSimulation(): void {
    this.isPaused = false;
    this.curFrameCountbyLastSecond = 0;
    this.fpsHeartbeat();
    this.iterate();
  }

  private fpsHeartbeat(): void {
    setTimeout(() => {
      this.curFrameCountbyLastSecond = 0;
      if (this.curFrameCountbyLastSecond >= this.fpsLimit) {
        this.startSimulation();
      } else {
        this.fpsHeartbeat();
      }
    }, 1000);
  }

  pauseSimulation(): void {
    this.isPaused = true;
  }

  private async init(
    modelPath: string,
    gridSize: [number, number],
    batchSize: number,
    channelSize: number,
    outputChannelSize: number,
    backend: string,
  ): Promise<void> {
    console.log('init called');
    const metaUrl = new URL(import.meta.url);
    console.log('metaUrl', metaUrl);
    // only keep the path part
    // specify the wasm path for onnxruntime-web, because the website could not find the wasm file installed by npm
    ort.env.wasm.wasmPaths = metaUrl.protocol + '//' + metaUrl.host + '/';
    this.session = await ort.InferenceSession.create(modelPath, {
      executionProviders: [backend],
      graphOptimizationLevel: 'all',
    });
    console.log('init session created');
    this.channelSize = channelSize;
    this.outputChannelSize = outputChannelSize;
    this.gridSize = gridSize;
    this.batchSize = batchSize;
    this.tensorShape = [batchSize, gridSize[0], gridSize[1], channelSize];
    this.tensorSize = batchSize * gridSize[0] * gridSize[1] * channelSize;
    this.outputSize = batchSize * gridSize[0] * gridSize[1] * outputChannelSize;
  }

  loadDataArray(data: number[][][][]): void {
    console.log(
      '🚀 ~ file: modelService.ts:132 ~ ModelService ~  initMatrixFromJSON ~ data:',
      data,
    );
    this.matrixArray = new Float32Array(data.flat(3));
    this.normalizeMatrix(this.matrixArray);
    if (this.matrixArray.length !== this.tensorSize) {
      throw new Error(
        `matrixArray length ${this.matrixArray.length} does not match tensorSize ${this.tensorSize}`,
      );
    }
    this.matrixArray = this.matrixMap(this.matrixArray, [0, 1], (v) =>
      Math.max(v, 0),
    );
    this.mass = this.matrixSum(this.matrixArray, [0, 1]);
  }

  private iterate(): void {
    if (this.session == null) {
      throw new Error(
        'session is null, createModelServices() must be called at first',
      );
    }
    console.log('iterate called');
    console.log('this.matrixArray', this.matrixArray);
    const inputEnergy = this.matrixSum(
      this.matrixArray,
      [1, 5],
      (value) => value ** 2,
    );
    const inputTensor = new ort.Tensor(
      'float32',
      this.matrixArray,
      this.tensorShape,
    );
    const feeds: Record<string, ort.Tensor> = {};
    feeds[this.session.inputNames[0]] = inputTensor;
    console.log(feeds);
    this.session
      .run(feeds)
      .then((outputs) => {
        // check if the output canbe downcasted to Float32Array
        if (
          outputs[this.session!.outputNames[0]] === undefined ||
          !(outputs[this.session!.outputNames[0]].data instanceof Float32Array)
        ) {
          throw new Error('output data is not Float32Array');
        }

        const outputData = this.constrainOutput(
          outputs[this.session!.outputNames[0]].data as Float32Array,
          inputEnergy,
        );
        this.outputCallback(outputData);
        this.curFrameCountbyLastSecond++;
        console.log(
          'curFrameCountbyLastSecond',
          this.curFrameCountbyLastSecond,
        );
        this.copyOutputToMatrix(outputData);
        setTimeout(() => {
          if (!this.isPaused) {
            if (this.curFrameCountbyLastSecond > this.fpsLimit) {
              this.isPaused = true;
              console.log(
                'fps limit reached, pause simulation, fpsLimit:',
                this.fpsLimit,
                'curFrameCountbyLastSecond:',
                this.curFrameCountbyLastSecond,
              );
            } else {
              this.iterate();
            }
          }
        });
      })
      .catch((e) => {
        console.error('error in session.run', e);
        this.isPaused = true;
      });
  }

  private normalizeMatrix(matrix: Float32Array): void {
    console.log('normalizeMatrix called');
    for (let i = 0; i < this.channelSize; i++) {
      matrix = this.normalizeMatrixChannel(matrix, i);
    }
  }

  private normalizeMatrixChannel(
    matrix: Float32Array,
    channel: number,
  ): Float32Array {
    const sum = this.matrixSum(
      matrix,
      [channel, channel + 1],
      (value) => value,
    );
    const mean = this.roundFloat(
      sum / (this.gridSize[0] * this.gridSize[1] * this.batchSize),
      4,
    );
    const std = this.roundFloat(
      Math.sqrt(
        this.matrixSum(matrix, [channel, channel + 1], (value) =>
          Math.pow(value - mean, 2),
        ) /
          (this.gridSize[0] * this.gridSize[1] * this.batchSize),
      ),
      4,
    );
    console.log('normalizeMatrixChannel', channel, mean, std);
    return this.matrixMap(
      matrix,
      [channel, channel + 1],
      (value) => (value - mean) / std,
    );
  }

  private constrainOutput(
    data: Float32Array,
    inputEnergy: number,
  ): Float32Array {
    let processed = this.constrainDensity(data);
    processed = this.constrainVelocity(processed, inputEnergy);
    return processed;
  }

  // data has cut off negative values (argument changed!)
  private constrainDensity(data: Float32Array): Float32Array {
    data = this.matrixMap(data, [0, 1], (value) => Math.max(value, 0), true);
    const sum = this.matrixSum(data, [0, 1], (value) => value, true);
    const scale = this.mass / sum;
    console.log(
      'Scaling density, cur mass:',
      sum,
      'target mass:',
      this.mass,
      'scale:',
      scale,
    );
    return this.matrixMap(data, [0, 1], (value) => value * scale, true);
  }

  private constrainVelocity(
    data: Float32Array,
    inputEnergy: number,
  ): Float32Array {
    const curEnergy = this.matrixSum(data, [1, 3], (value) => value ** 2, true);
    const scale = this.roundFloat(Math.sqrt(inputEnergy / curEnergy), 4);
    console.log(
      'Scaling velocity, cur energy:',
      curEnergy,
      'target energy:',
      inputEnergy,
      'scale:',
      scale,
    );
    if (scale >= 1) return data;
    return this.matrixMap(data, [1, 3], (value) => value * scale, true);
  }

  private copyOutputToMatrix(outputs: Float32Array): void {
    if (this.matrixArray.length === 0) {
      throw new Error('matrixArray is empty');
    }
    let fromIndex = 0;
    let toIndex = 0;
    let cntOffset = 0;
    while (fromIndex < outputs.length) {
      if (cntOffset >= 3) {
        cntOffset = 0;
        toIndex += 2;
        if (toIndex >= this.matrixArray.length) {
          throw new Error(
            `toIndex ${toIndex} exceeds matrixArray length ${this.matrixArray.length}`,
          );
        }
      }
      this.matrixArray[toIndex] = outputs[fromIndex];
      fromIndex++;
      toIndex++;
      cntOffset++;
    }
    if (fromIndex !== outputs.length) {
      throw new Error(
        `fromIndex ${fromIndex} does not match outputs length ${outputs.length}`,
      );
    }
    if (toIndex + 2 !== this.matrixArray.length) {
      throw new Error(
        `toIndex ${toIndex} does not match matrixArray length ${this.matrixArray.length}`,
      );
    }
  }

  updateForce(pos: Vector2, forceDelta: Vector2): void {
    const index: number = this.getIndex(pos);
    this.matrixArray[index + 3] += forceDelta.x;
    this.matrixArray[index + 4] += forceDelta.y;
  }

  private getIndex(pos: Vector2, batchIndex = 0): number {
    return (
      batchIndex * this.gridSize[0] * this.gridSize[1] * this.channelSize +
      pos.y * this.gridSize[1] * this.channelSize +
      pos.x * this.channelSize
    );
  }

  /**
   * Calculate the sum of a matrix Array with provided transform applied on matrix values.
   * @param matrix input matrix Array
   * @param channelRange range of selected channel
   * @param f transform function for matrix values
   * @param isOutput is the matrix from model output
   * @returns sum of the selected channel of matrix
   */
  private matrixSum(
    matrix: Float32Array,
    channelRange: [number, number],
    f: (value: number) => number = (value) => value,
    isOutput = false,
  ): number {
    const tensorSize = isOutput ? this.outputSize : this.tensorSize;
    const channelSize = isOutput ? this.outputChannelSize : this.channelSize;
    let sum = 0;
    let index = 0;
    while (index < tensorSize) {
      for (let k = channelRange[0]; k < channelRange[1]; k++) {
        sum += f(matrix[index + k]);
      }
      index += channelSize;
    }
    return sum;
  }

  /**
   * Map the transform function to selected channel of input matrix.
   * @param matrix input matrix Array
   * @param channelRange range of selected channel
   * @param f transform function for matrix values
   * @param isOutput is the matrix from model output
   * @returns sum of the selected channel of matrix
   */
  private matrixMap(
    matrix: Float32Array,
    channelRange: [number, number],
    f: (value: number) => number,
    isOutput = false,
  ): Float32Array {
    const tensorSize = isOutput ? this.outputSize : this.tensorSize;
    const channelSize = isOutput ? this.outputChannelSize : this.channelSize;
    let index = 0;
    while (index < tensorSize) {
      for (let k = channelRange[0]; k < channelRange[1]; k++) {
        matrix[index + k] = f(matrix[index + k]);
      }
      index += channelSize;
    }
    return matrix;
  }

  private roundFloat(value: number, decimal = 4): number {
    return Math.round(value * 10 ** decimal) / 10 ** decimal;
  }

  getInputTensor(): Float32Array {
    return this.matrixArray;
  }

  getMass(): number {
    return this.mass;
  }

  getInputShape(): [number, number, number, number] {
    return this.tensorShape;
  }

  setMass(mass: number): void {
    this.mass = mass;
  }

  getType(): string {
    return 'onnx';
  }
}
