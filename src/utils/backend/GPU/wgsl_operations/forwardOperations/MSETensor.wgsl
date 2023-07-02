fn MSETensor(gId : vec3u, t : u32){
    let curr_m = u32(ping.entries[u32(offset.tensor[t].rows)]);
    let curr_n = u32(ping.entries[u32(offset.tensor[t].cols)]);
    let curr_DataFirstIndex = u32(offset.tensor[t].data);
    let curr_ParentTrue = u32(ping.entries[u32(offset.tensor[t].parents)]);
    let curr_ParentPred = u32(ping.entries[u32(offset.tensor[t].parents) + 1u]);
    let parentTrue_DataFirstIndex = u32(offset.tensor[curr_ParentTrue].data);
    let parentPred_DataFirstIndex = u32(offset.tensor[curr_ParentPred].data);
    let parentTrue_m = u32(ping.entries[u32(offset.tensor[curr_ParentTrue].rows)]);
    let parentTrue_n = u32(ping.entries[u32(offset.tensor[curr_ParentTrue].cols)]);

    if (gId.x >= u32(1) || gId.y >= u32(1)) {
        return; // Guard against out-of-bounds work group sizes
    }

    var mse = f32(0.0);
    for (var i = 0u; i < parentTrue_m; i = i + 1u) {
        for (var j = 0u; j < parentTrue_n; j = j + 1u) {
            let indexTrue = parentTrue_DataFirstIndex + i * parentTrue_n + j;
            let indexPred = parentPred_DataFirstIndex + i * parentTrue_n + j;
            mse += pow(ping.entries[indexPred] - ping.entries[indexTrue] , f32(2));
        }
    }
    mse = mse / ( f32(parentTrue_n) * f32(parentTrue_m));

    let index = gId.y + gId.x * curr_n;
    ping.entries[curr_DataFirstIndex + index] = mse;

    let currIteration = u32(control.iteration);
    accuracies[currIteration] = mse;


    // clean up the gradient data
    let curr_GradientData = u32(offset.tensor[t].gradientData);
    ping.entries[curr_GradientData + index] = f32(0);
}