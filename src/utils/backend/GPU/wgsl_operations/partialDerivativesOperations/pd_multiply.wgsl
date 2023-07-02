

// fn pdLeft_multiply(gId : vec3u, t : u32){
//     let curr_m = u32(ping.entries[u32(offset.tensor[t].rows)]);
//     let curr_n = u32(ping.entries[u32(offset.tensor[t].cols)]);
//     let curr_ParentLeft = u32(ping.entries[u32(offset.tensor[t].parents)]);
//     let curr_ParentRight = u32(ping.entries[u32(offset.tensor[t].parents) + 1u]);
//     let parentRight_DataFirstIndex = u32(offset.tensor[curr_ParentRight].data);
//     let parentLeft_m = u32(ping.entries[u32(offset.tensor[curr_ParentLeft].rows)]);
//     let parentLeft_n = u32(ping.entries[u32(offset.tensor[curr_ParentLeft].cols)]);
//     let parentRight_m = u32(ping.entries[u32(offset.tensor[curr_ParentRight].rows)]);
//     let parentRight_n = u32(ping.entries[u32(offset.tensor[curr_ParentRight].cols)]);
//     let curr_pdLeft = u32(offset.tensor[t].partialDerivativeLeft);

//     if ( gId.x >= parentRight_m || gId.y >= parentRight_n){
//         return; // Guard against out-of-bounds work group sizes
//     }

//     // ~ ((1.0 / (double) n) * right.getData());
//     let indexParRight = parentRight_DataFirstIndex + gId.y + gId.x * parentRight_n;
//     let indexPdLeft = curr_pdLeft + gId.y + gId.x * parentRight_n;
//     ping.entries[indexPdLeft] = ping.entries[indexParRight] / f32(curr_n);        
// }

// fn pdRight_multiply(gId : vec3u, t : u32){
//     let curr_m = u32(ping.entries[u32(offset.tensor[t].rows)]);
//     let curr_n = u32(ping.entries[u32(offset.tensor[t].cols)]);
//     let curr_ParentLeft = u32(ping.entries[u32(offset.tensor[t].parents)]);
//     let curr_ParentRight = u32(ping.entries[u32(offset.tensor[t].parents) + 1u]);
//     let parentLeft_DataFirstIndex = u32(offset.tensor[curr_ParentLeft].data);
//     let parentLeft_m = u32(ping.entries[u32(offset.tensor[curr_ParentLeft].rows)]);
//     let parentLeft_n = u32(ping.entries[u32(offset.tensor[curr_ParentLeft].cols)]);
//     let parentRight_m = u32(ping.entries[u32(offset.tensor[curr_ParentRight].rows)]);
//     let parentRight_n = u32(ping.entries[u32(offset.tensor[curr_ParentRight].cols)]);
//     let curr_pdRight = u32(offset.tensor[t].partialDerivativeRight);

//     if ( gId.x >= parentLeft_m || gId.y >= parentLeft_n){
//         return; // Guard against out-of-bounds work group sizes
//     }

//     //  ~ ((1.0 / (double) n) * left.getData()); ???
//     let indexParLeft = parentLeft_DataFirstIndex + gId.y + gId.x * parentLeft_n;
//     let indexPdRight = curr_pdRight + gId.y + gId.x * parentLeft_n;
//     ping.entries[indexPdRight] = ping.entries[indexParLeft]; // / f32(curr_n);
// }

// fn pd_multiply(gId : vec3u, t : u32, currParentId : u32){
//     let isRightMultiplicator = u32(ping.entries[u32(offset.tensor[currParentId].isRightMultiplicator)]);
//     if(isRightMultiplicator == u32(0)){
//         pdLeft_multiply(gId, t);
//     }
//     else{
//         pdRight_multiply(gId, t);
//     }
// }


// --------------------------------------------------------------------------------------------------------



// fn pdLeft_multiply(gId : vec3u, t : u32){
//     let curr_m = u32(ping.entries[u32(offset.tensor[t].rows)]);
//     let curr_n = u32(ping.entries[u32(offset.tensor[t].cols)]);
//     let curr_ParentLeft = u32(ping.entries[u32(offset.tensor[t].parents)]);
//     let curr_ParentRight = u32(ping.entries[u32(offset.tensor[t].parents) + 1u]);
//     let parentRight_DataFirstIndex = u32(offset.tensor[curr_ParentRight].data);
//     let parentLeft_m = u32(ping.entries[u32(offset.tensor[curr_ParentLeft].rows)]);
//     let parentLeft_n = u32(ping.entries[u32(offset.tensor[curr_ParentLeft].cols)]);
//     let parentRight_m = u32(ping.entries[u32(offset.tensor[curr_ParentRight].rows)]);
//     let parentRight_n = u32(ping.entries[u32(offset.tensor[curr_ParentRight].cols)]);
//     let curr_pdLeft = u32(offset.tensor[t].partialDerivativeLeft);

//     if ( gId.x >= parentRight_m || gId.y >= parentRight_n){
//         return; // Guard against out-of-bounds work group sizes
//     }

//     // ~ trans((1.0 / (double) n) * right.getData());
//     let indexParRight = parentRight_DataFirstIndex + gId.y + gId.x * parentRight_n;
//     let indexPdLeft = curr_pdLeft + gId.x + gId.y * parentRight_m;
//     ping.entries[indexPdLeft] = ping.entries[indexParRight] / f32(curr_n);          
// }

// fn pdRight_multiply(gId : vec3u, t : u32){
//     let curr_m = u32(ping.entries[u32(offset.tensor[t].rows)]);
//     let curr_n = u32(ping.entries[u32(offset.tensor[t].cols)]);
//     let curr_ParentLeft = u32(ping.entries[u32(offset.tensor[t].parents)]);
//     let curr_ParentRight = u32(ping.entries[u32(offset.tensor[t].parents) + 1u]);
//     let parentLeft_DataFirstIndex = u32(offset.tensor[curr_ParentLeft].data);
//     let parentLeft_m = u32(ping.entries[u32(offset.tensor[curr_ParentLeft].rows)]);
//     let parentLeft_n = u32(ping.entries[u32(offset.tensor[curr_ParentLeft].cols)]);
//     let parentRight_m = u32(ping.entries[u32(offset.tensor[curr_ParentRight].rows)]);
//     let parentRight_n = u32(ping.entries[u32(offset.tensor[curr_ParentRight].cols)]);
//     let curr_pdRight = u32(offset.tensor[t].partialDerivativeRight);

//     if ( gId.x >= parentLeft_m || gId.y >= parentLeft_n){
//         return; // Guard against out-of-bounds work group sizes
//     }

//     //  ~ trans((1.0 / (double) n) * left.getData());
//     let indexParLeft = parentLeft_DataFirstIndex + gId.y + gId.x * parentLeft_n;
//     let indexPdRight = curr_pdRight  + gId.x + gId.y * parentLeft_m;
//     ping.entries[indexPdRight] = ping.entries[indexParLeft]; // / f32(curr_n);
// }


// ------------------------------------------------------------------------------------------------------------

// copy the data of the parent_partner in the current parents data
fn pd_multiply(gId : vec3u, t : u32, currParentId : u32){
    // let parent_partner_m = u32(ping.entries[u32(offset.tensor[currParentId].partner_rows)]);
    // let parent_partner_n = u32(ping.entries[u32(offset.tensor[currParentId].partner_cols)]);

    
    // if ( gId.x >= parent_partner_m || gId.y >= parent_partner_n){
    //     return; // Guard against out-of-bounds work group sizes
    // }

    // let parent_partner_id = u32(ping.entries[u32(offset.tensor[currParentId].partner_id)]);
    // let parent_partner_DataFirstIndex = u32(offset.tensor[parent_partner_id].data);
    // let isRightMultiplicator = u32(ping.entries[u32(offset.tensor[currParentId].isRightMultiplicator)]);

    // if(isRightMultiplicator == u32(0)){
    //     let indexPd = u32(offset.tensor[t].partialDerivativeLeft);
    //     let indexPartnerData = parent_partner_DataFirstIndex + gId.x + gId.y * parent_partner_n;
    //     let index = indexPd + gId.x + gId.y * parent_partner_n;
    //     ping.entries[index] = ping.entries[indexPartnerData] / f32(parent_partner_n);
    // }
    // else{
    //     let indexPd = u32(offset.tensor[t].partialDerivativeRight);
    //     let indexPartnerData = parent_partner_DataFirstIndex + gId.x + gId.y * parent_partner_n;
    //     let index = indexPd + gId.x + gId.y * parent_partner_n;
    //     ping.entries[index] = ping.entries[indexPartnerData];
    // }
}
