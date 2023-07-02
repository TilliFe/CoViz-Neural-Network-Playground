import { MatMul } from '../../GPU/initModel/GPUtraining';
import Data from '../tools/DataClass';
import { Model } from '../tools/ModelVisual';
import { Tensor } from '../tools/TensorClass';

function setUpModel(
  setAvgError,
  setEdgesActive,
  setXVals,
  setTrueVals,
  setPredVals,
  data: Data,
  tensors,
  iterations,
  learningRate,
  momentum,
  batchSize,
  stopLearning,
  setBreakTraining
) {
  if (tensors.length == 0) {
    return;
  }      
   

  
  const model = new Model();
  model.numTensors = tensors.length;

  for (let i = 0; i < tensors.length; ++i) {
    const type = tensors[i].type;
    const tensorId = tensors[i].id;
    const rows = tensors[i].rows;
    const cols = tensors[i].cols;
    const requiresGradient = tensors[i].requiresGradient;
    const tensor = new Tensor(type, rows, cols, requiresGradient, tensorId);
    const initialization = tensors[i].initialization;

    tensor.parents = tensors[i].parents;
    tensor.children = tensors[i].children;
    tensor.isRightMultiplicator = tensors[i].isRightMultiplicator;
      
    
    if (initialization == 'random' && type == 'none') {
      tensor.setRandomData();
    } else if (initialization == 'ones' && type == 'none') {
      tensor.setAllOnes();
    }
    if (tensors[i].isLast || type == 'MSE' || type == 'CE') {
      model.lastTensor = tensorId;
    }
    if (tensors[i].isInput) {
      tensor.isInput = true;
      data.setInputTensor(tensor);
      tensor.requiresGradient = 0;
      model.inputTensor = tensorId;
    }
    if (tensors[i].isTrue) {
      tensor.isTrue = true;
      data.setTrueTensor(tensor);
      model.trueTensor = tensorId;
      tensor.requiresGradient = 1;
    }
    if (tensors[i].isOutput) {
      model.outputTensor = tensorId;
    }
    model.tensors.push(tensor);
  }

  const maxNumIterations = iterations;
  model.learningRate = learningRate;
  model.momentum = momentum;
  model.batchSize = batchSize;

  // find largest m or n
  let maxBlockSize: number;
  maxBlockSize = 0;
  for (let i = 0; i < model.numTensors; i++) {
    if (model.tensors[i].rows > maxBlockSize) {
      maxBlockSize = model.tensors[i].rows;
    }
    if (model.tensors[i].cols > maxBlockSize) {
      maxBlockSize = model.tensors[i].cols;
    }
  }

  // init partial Derivatives to zero with proper lengths
  for (let i = 0; i < model.numTensors; i++) {
    // copy
    if (model.tensors[i].type == 0) {
      model.tensors[i].partialDerivativeLeft = new Array<number>(1).fill(0);
      model.tensors[i].partialDerivativeRight = new Array<number>(1).fill(0);
      // continue;
    }

    // add
    else if (model.tensors[i].type == 1) {
      const par1 = model.tensors[i].parents[0];
      const par2 = model.tensors[i].parents[1];
      const m = model.tensors[par1].rows;
      const n = model.tensors[par1].cols;
      model.tensors[i].partialDerivativeLeft = new Array<number>(1).fill(0);
      model.tensors[i].partialDerivativeRight = new Array<number>(1).fill(0);
      model.tensors[par1].isRightMultiplicator = 0;
      model.tensors[par2].isRightMultiplicator = 1;
      model.tensors[par1].partner_rows = m;
      model.tensors[par1].partner_cols = n;
      model.tensors[par1].partner_id = par2;
      model.tensors[par2].partner_rows = m;
      model.tensors[par2].partner_cols = n;
      model.tensors[par2].partner_id = par1;
    }

    // multiply
    else if (model.tensors[i].type == 2) {
      const par1 = model.tensors[i].parents[0];
      const par2 = model.tensors[i].parents[1];
      const m = model.tensors[par1].rows;
      const n = model.tensors[par1].cols; // ( = model.tensors[par2].rows  )
      const k = model.tensors[par2].cols;
      model.tensors[i].partialDerivativeLeft = new Array<number>(1).fill(0);
      model.tensors[i].partialDerivativeRight = new Array<number>(1).fill(0);
      model.tensors[par1].partner_rows = n;
      model.tensors[par1].partner_cols = k;
      model.tensors[par1].partner_id = par2;
      model.tensors[par2].partner_rows = m;
      model.tensors[par2].partner_cols = n;
      model.tensors[par2].partner_id = par1;
      model.tensors[par1].isRightMultiplicator = 0;
      model.tensors[par2].isRightMultiplicator = 1;
    }


    // ReLU
    else if (model.tensors[i].type == 3) {
      const par1 = model.tensors[i].parents[0];
      const m = model.tensors[par1].rows;
      const n = model.tensors[par1].cols;
      model.tensors[i].partialDerivativeLeft = new Array<number>(1).fill(0);
      model.tensors[i].partialDerivativeRight = new Array<number>(1).fill(0);
      // model.tensors[par1].isRightMultiplicator = 0;
    }

    // softmax
    else if (model.tensors[i].type == 4) {
      const par1 = model.tensors[i].parents[0];
      const m = model.tensors[par1].rows;
      const n = model.tensors[par1].cols;
      model.tensors[i].partialDerivativeLeft = new Array<number>(1).fill(0);
      model.tensors[i].partialDerivativeRight = new Array<number>(1).fill(0);
      // model.tensors[par1].isRightMultiplicator = 0;
    }

    // CE
    else if (model.tensors[i].type == 5) {
      const par1 = model.tensors[i].parents[0];
      const par2 = model.tensors[i].parents[1];
      const m = model.tensors[par1].rows;
      const n = model.tensors[par1].cols;
      model.tensors[i].partialDerivativeLeft = new Array<number>(1).fill(0);
      model.tensors[i].partialDerivativeRight = new Array<number>(1).fill(0);
      model.tensors[par1].partner_rows = m;
      model.tensors[par1].partner_cols = n;
      model.tensors[par1].partner_id = par2;
      model.tensors[par2].partner_rows = m;
      model.tensors[par2].partner_cols = n;
      model.tensors[par2].partner_id = par1;
    }

    // OneHot
    else if (model.tensors[i].type == 6) {
    }

    // MSE
    else if (model.tensors[i].type == 7) {
      const par1 = model.tensors[i].parents[0];
      const par2 = model.tensors[i].parents[1];
      const m = model.tensors[par1].rows;
      const n = model.tensors[par1].cols;
      model.tensors[i].partialDerivativeLeft = new Array<number>(1).fill(0);
      model.tensors[i].partialDerivativeRight = new Array<number>(1).fill(0);
      model.tensors[par1].partner_id = par2;
      model.tensors[par2].partner_id = par1;
      model.tensors[par1].isRightMultiplicator = 0;
      model.tensors[par2].isRightMultiplicator = 1;
    }

    // conv2d
    else if (model.tensors[i].type == 8) {
      // const par1 = model.tensors[i].parents[0]; // is X
      // const par2 = model.tensors[i].parents[1]; // is kernel
      // const m = model.tensors[par1].rows;
      // const n = model.tensors[par1].cols;
      // const m_k = model.tensors[par2].rows;
      // const n_k = model.tensors[par2].cols;
      // model.tensors[i].partialDerivativeLeft = new Array<Number>(1).fill(0);
      // model.tensors[i].partialDerivativeRight = new Array<Number>(1).fill(0);
    }
  }

  // console.log(model);

  let flatData = [];
  const offsets = []; // where does each tensor start in the flattened array
  let tensorOffsets = [];
  let offset = 0;

  tensorOffsets.push(model.numTensors);
  tensorOffsets.push(0); // set to total number of floats after completely filled
  tensorOffsets.push(maxBlockSize);

  for (let i = 0; i < model.numTensors; i++) {
    offsets.push(offset); // tensor i starts at index offset in flatData

    flatData.push(model.tensors[i].type);
    flatData.push(model.tensors[i].isRightMultiplicator);
    flatData.push(model.tensors[i].requiresGradient);
    flatData.push(model.tensors[i].rows);
    flatData.push(model.tensors[i].cols);
    flatData.push(model.tensors[i].partner_rows);
    flatData.push(model.tensors[i].partner_cols);
    flatData.push(model.tensors[i].partner_id);
    flatData = flatData.concat(Array.from(model.tensors[i].data));
    flatData = flatData.concat(Array.from(model.tensors[i].gradientData));
    flatData = flatData.concat(Array.from(model.tensors[i].velocity_momentum));
    flatData = flatData.concat(Array.from(model.tensors[i].velocity_RMSProp));
    flatData = flatData.concat(Array.from(model.tensors[i].children));
    flatData = flatData.concat(Array.from(model.tensors[i].parents));
    flatData = flatData.concat(
      Array.from(model.tensors[i].partialDerivativeLeft)
    );
    flatData = flatData.concat(
      Array.from(model.tensors[i].partialDerivativeRight)
    );
    flatData.push(model.tensors[i].metaDims.length);
    flatData = flatData.concat(Array.from(model.tensors[i].metaDims));



    // if (i == data.tensorInputId) {
    //   data.inputOffset = offset + 6;
    // }
    // if (i == data.tensorTrueId) {
    //   data.trueOffset = offset + 6;
    // }


    
    tensorOffsets = tensorOffsets.concat([
      offset,
      ++offset,
      ++offset,
      ++offset,
      ++offset, // m (rows)
      ++offset, // n (cols)
      ++offset, //  (partner_rows)
      ++offset, //  (partner_cols)
      ++offset, //  (partner_id)
      offset + model.tensors[i].data.length,
      offset +
        model.tensors[i].data.length +
        model.tensors[i].gradientData.length,
      offset +
        model.tensors[i].data.length +
        model.tensors[i].gradientData.length +
        model.tensors[i].velocity_momentum.length,
      offset +
        model.tensors[i].data.length +
        model.tensors[i].gradientData.length +
        model.tensors[i].velocity_momentum.length * 2, // times 2 because we have the velocity_RMSProp as well
      offset +
        model.tensors[i].data.length +
        model.tensors[i].gradientData.length +
        model.tensors[i].velocity_momentum.length * 2 +
        model.tensors[i].children.length,
      offset +
        model.tensors[i].data.length +
        model.tensors[i].gradientData.length +
        model.tensors[i].velocity_momentum.length * 2 +
        model.tensors[i].children.length +
        model.tensors[i].parents.length,
      offset +
        model.tensors[i].data.length +
        model.tensors[i].gradientData.length +
        model.tensors[i].velocity_momentum.length * 2 +
        model.tensors[i].children.length +
        model.tensors[i].parents.length +
        model.tensors[i].partialDerivativeLeft.length,
      offset +
        model.tensors[i].data.length +
        model.tensors[i].gradientData.length +
        model.tensors[i].velocity_momentum.length * 2 +
        model.tensors[i].children.length +
        model.tensors[i].parents.length +
        model.tensors[i].partialDerivativeLeft.length +
        model.tensors[i].partialDerivativeRight.length,
      offset +
        model.tensors[i].data.length +
        model.tensors[i].gradientData.length +
        model.tensors[i].velocity_momentum.length * 2 +
        model.tensors[i].children.length +
        model.tensors[i].parents.length +
        model.tensors[i].partialDerivativeLeft.length +
        model.tensors[i].partialDerivativeRight.length +
        1,
    ]);
    offset =
      offset +
      model.tensors[i].data.length +
      model.tensors[i].gradientData.length +
      model.tensors[i].velocity_momentum.length * 2 +
      model.tensors[i].children.length +
      model.tensors[i].parents.length +
      model.tensors[i].partialDerivativeLeft.length +
      model.tensors[i].partialDerivativeRight.length +
      1 +
      model.tensors[i].metaDims.length;
  }

  tensorOffsets[1] = flatData.length;

  // build sequence for the backwards pass for computing the partialDerivatives
  let queueCount = [];
  for (let i = 0; i < model.numTensors; i++) {
    queueCount.push(0);
  }
  const backwardTape = [];
  backwardTape.push(0);
  backwardTape.push(model.learningRate);
  backwardTape.push(model.momentum);

  let queue = [model.lastTensor];
  while (queue.length > 0) {
    const currentVertex = queue.shift();
    const parents = model.tensors[currentVertex].parents;
    backwardTape.push(currentVertex);
    for (const par of parents) {
      if (model.tensors[par].requiresGradient) {
        queueCount[par]++;
        if (queueCount[par] == model.tensors[par].children.length) {
          queue.push(par);
        }
      }
    }
  }
  backwardTape[0] = model.numTensors;

  // build sequence for the backwards pass for computing the gradients
  queueCount = [];
  for (let i = 0; i < model.numTensors; i++) {
    queueCount.push(0);
  }
  const gradientTape = [];
  queue = [model.lastTensor];
  while (queue.length > 0) {
    const currentVertex = queue.shift();
    const parents = model.tensors[currentVertex].parents;
    for (const par of parents) {
      if (model.tensors[par].requiresGradient) {
        queueCount[par]++;
        gradientTape.push(par);
        gradientTape.push(currentVertex);
        if (queueCount[par] == model.tensors[par].children.length) {
          queue.push(par);
        }
      }
    }
  }

  // build sequence for the forwards pass
  queueCount = [];
  for (let i = 0; i < model.numTensors; i++) {
    queueCount.push(0);
  }
  const forwardTape = [];
  queue = [model.lastTensor];
  while (queue.length > 0) {
    const currentVertex = queue.shift();
    const parents = model.tensors[currentVertex].parents;
    forwardTape.push(currentVertex);
    for (const par of parents) {
      if (model.tensors[par].type != 0) {
        queueCount[par]++;
        if (queueCount[par] == model.tensors[par].children.length) {
          queue.push(par);
        }
      }
    }
  }
  forwardTape.reverse();
  console.log(forwardTape);
  console.log(backwardTape);
  console.log(gradientTape);

  // // convert everything to the proper data type. here: f32
  const f32TensorOffsets = new Float32Array(tensorOffsets);
  const f32FlatData = new Float32Array(flatData);
  const f32BackwardTape = new Float32Array(backwardTape);
  const f32GradientTape = new Float32Array(gradientTape);

  for (let i = 0; i < gradientTape.length; i = i + 2) {
    const curr = model.tensors[gradientTape[i]];
    const child = model.tensors[gradientTape[i + 1]];
    console.log(' ');
    console.log(curr.typestring + ' -> ' + child.typestring);
    console.log('curr is rightmultiplicator: ' + curr.isRightMultiplicator);
  }

  console.log(model);
  MatMul(
    setAvgError,
    setEdgesActive,
    setXVals,
    setTrueVals,
    setPredVals,
    f32TensorOffsets,
    f32FlatData,
    f32BackwardTape,
    f32GradientTape,
    maxNumIterations,
    data,
    model,
    forwardTape,
    gradientTape,
    backwardTape,
    stopLearning,
    setBreakTraining
  );
  return <></>;
}

export default setUpModel;
