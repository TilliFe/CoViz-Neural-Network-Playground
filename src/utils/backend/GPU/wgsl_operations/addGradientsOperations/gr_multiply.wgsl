fn gr_multiply(gId : vec3u, currTensor : u32, childTensor: u32){

    let curr_m = u32(ping.entries[u32(offset.tensor[currTensor].rows)]);
    let curr_n = u32(ping.entries[u32(offset.tensor[currTensor].cols)]);

    let childGradient_m = u32(ping.entries[u32(offset.tensor[childTensor].rows)]); 
    let childGradient_n = u32(ping.entries[u32(offset.tensor[childTensor].cols)]);

    let curr_partner_id = u32(ping.entries[u32(offset.tensor[currTensor].partner_id)]);
    let curr_partner_data = u32(ping.entries[u32(offset.tensor[curr_partner_id].data)]);

    let partner_rows = u32(ping.entries[u32(offset.tensor[currTensor].partner_rows)]);
    let partner_cols = u32(ping.entries[u32(offset.tensor[currTensor].partner_cols)]);

    if (gId.x >= curr_m || gId.y >= curr_n) {
        return; // Guard against out-of-bounds work group sizes
    }
    
    let curr_GradientDataFirst = u32(offset.tensor[currTensor].gradientData);
    let child_GradientDataFirst = u32(offset.tensor[childTensor].gradientData);

    let isRightMultiplicator = u32(ping.entries[u32(offset.tensor[currTensor].isRightMultiplicator)]);


    if(isRightMultiplicator == u32(0)){
        // compute gradient_child x partner_transpose 
        var result = f32(0.0);
        for (var i = 0u; i < partner_cols; i = i + 1u) {
            let a = child_GradientDataFirst + gId.x * childGradient_n + i;
            let b = curr_partner_data + gId.y * partner_cols + i;
            result += ping.entries[a] * ping.entries[b];
        }
        let index = curr_GradientDataFirst + gId.x * partner_rows + gId.y;
        ping.entries[index] = result / f32(childGradient_n);        
    } 
    if(isRightMultiplicator == u32(1)){
        // compute partner_transpose x gradient_child
        var result = f32(0.0);
        for (var i = 0u; i < partner_rows; i = i + 1u) {
            let a = curr_partner_data + i * partner_cols + gId.x;
            let b = child_GradientDataFirst + i * childGradient_n + gId.y;
            result += ping.entries[a] * ping.entries[b];
        }
        let index = curr_GradientDataFirst + gId.x * childGradient_n + gId.y;
        ping.entries[index] = result;
    }
    
}


