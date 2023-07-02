fn gr_softmaxCE(gId : vec3u, currTensor : u32, childTensor: u32){

    let curr_m = u32(ping.entries[u32(offset.tensor[currTensor].rows)]);    
    let curr_n = u32(ping.entries[u32(offset.tensor[currTensor].cols)]);

    if (gId.x >= curr_m || gId.y >= curr_n) {
        return; // Guard against out-of-bounds work group sizes
    } 

    let child_partner_id = u32(ping.entries[u32(offset.tensor[childTensor].partner_id)]);
    let child_partner_data = u32(ping.entries[u32(offset.tensor[child_partner_id].data)]);

    let child_Data = u32(offset.tensor[currTensor].data);
    let curr_GradientData = u32(offset.tensor[currTensor].gradientData);

    let index = gId.y + gId.x * curr_n;
    ping.entries[curr_GradientData + index] = (ping.entries[child_Data + index] - ping.entries[child_partner_data + index]);
}