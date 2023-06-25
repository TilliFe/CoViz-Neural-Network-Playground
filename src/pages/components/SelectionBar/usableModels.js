import React, { useState } from 'react';
import Button from '@mui/material/Button';
import Stack from '@mui/material/Stack';
import AddIcon from '@mui/icons-material/Add';
import { Box } from '@mui/material';

import { useDispatch } from 'react-redux';
import { computeGraphActions } from '../../../store/ComputeGraph-slice';
import Typography from '@mui/material/Typography';
import GitHubIcon from '@mui/icons-material/GitHub';
import Grid from '@mui/material/Grid';

const buttonStyles = {
  height: '53px',
  borderRadius: '8px',
  backgroundColor: 'rgb(70,70,70)',
  color: 'rgb(255,255,255)',
  marginLeft: '10px',
  marginRight: '10px',
  marginTop: '10px',
  fontSize: '16px',

  '&:hover': {
    backgroundColor: 'rgb(130,130,130)',
  },
};

const SelectModel = () => {
  const [clickedButton, setClickedButton] = useState(1);

  const dispatch = useDispatch();
  const setPredefinedModel = (name) => {
    dispatch(computeGraphActions.setPredefinedModel(name));
  };
  const setClicked = (number) => {
    dispatch(computeGraphActions.setClicked(number));
  };
  const setEdgesActive = (name) => {
    dispatch(computeGraphActions.setEdgesActive(name));
  };

  const handleClick = (buttonNumber) => {
    setClickedButton(buttonNumber);
  };

  return (
    <Grid
      container
      direction="column"
      spacing={0}
      sx={{
        position: 'relative',
        height: '100%',
        overflow: 'auto',
        padding: 0,
        margin: 0,
        bgcolor: 'rgb(40,40,40)',
      }}
    >
      <Grid
        item
        sx={{ height: { xs: '50%', sm: '60%', md: '65%' }, overflow: 'auto' }}
      >
        <Stack style={{}} direction="column">
          <Button
            variant="contained"
            onClick={() => {
              handleClick(0);
              setPredefinedModel('');
              setClicked(0);
              setEdgesActive(false);
            }}
            sx={{
              ...buttonStyles,
              backgroundColor:
                clickedButton === 0
                  ? 'rgb(110,110,110)'
                  : buttonStyles.backgroundColor,
              border: 'solid 0px rgb(155,150,150)',
              height: '53px',
              paddingLeft: '25px',
            }}
            startIcon={<AddIcon />}
          >
            <Box
              sx={{
                display: 'flex',
                justifyContent: 'flex-start',
                width: '100%',
                paddingLeft: '3px',
              }}
            >
              New Model
            </Box>
          </Button>

          <Button
            variant="contained"
            onClick={() => {
              handleClick(1);
              setPredefinedModel('Regression1');
              setClicked(0);
              setEdgesActive(false);
            }}
            sx={{
              ...buttonStyles,
              backgroundColor:
                clickedButton === 1
                  ? 'rgb(110,110,110)'
                  : buttonStyles.backgroundColor,
            }}
          >
            <Box
              sx={{
                display: 'flex',
                justifyContent: 'flex-start',
                width: '100%',
                paddingLeft: '3px',
              }}
            >
              Regression Model 1
            </Box>
          </Button>
          <Button
            variant="contained"
            onClick={() => {
              handleClick(2);
              setPredefinedModel('Classification1');
            }}
            sx={{
              ...buttonStyles,
              backgroundColor:
                clickedButton === 2
                  ? 'rgb(110,110,110)'
                  : buttonStyles.backgroundColor,
            }}
          >
            <Box
              sx={{
                display: 'flex',
                justifyContent: 'flex-start',
                width: '100%',
                paddingLeft: '3px',
              }}
            >
              Classification Model 1
            </Box>
          </Button>
        </Stack>
      </Grid>
      <Grid
        item
        sx={{ height: { xs: '35%', sm: '30%', md: '25%' }, overflow: 'auto' }}
      >
        <Stack style={{}} direction="column">
          <Typography
            color="grey.500"
            padding="3px"
            margin="7px"
            fontSize="12px"
          >
            You need to enable{' '}
            <strong style={{ color: 'white' }}>WebGPU</strong> in your browser
            flags!
          </Typography>

          <Typography
            color="grey.500"
            padding="3px"
            margin="7px"
            fontSize="12px"
          >
            <strong>Add</strong> a Tensor Node by &apos;Right Click&apos; on the
            Canvas
          </Typography>

          <Typography
            color="grey.500"
            padding="3px"
            margin="7px"
            fontSize="12px"
          >
            <strong>Connect</strong> two Nodes by dragging from one Node to
            another
          </Typography>

          <Typography
            color="grey.500"
            padding="3px"
            margin="7px"
            fontSize="12px"
          >
            <strong>Remove</strong> any Tensor Node or Edge between Nodes by
            double &apos;Left Click&apos; on it.
          </Typography>

          <Typography color="grey.500" padding="3px" margin="7px"></Typography>
        </Stack>
      </Grid>

      <Grid item sx={{ height: { xs: '8%', sm: '5%', md: '3%' } }}>
        <Button
          onClick={() =>
            window.open(
              'https://github.com/TilliFe/CoViz-Neural-Network-Playground'
            )
          }
          variant="contained"
          sx={{
            position: 'relative',
            width: '270px',
            height: '48px',
            color: 'rgb(255,255,255)',
            backgroundColor: 'rgb(70,50,60)',
            borderRadius: '10px',
            fontSize: '15px',
            margin: '12px',
          }}
          startIcon={<GitHubIcon />}
        >
          GitHub Docs
        </Button>
      </Grid>
    </Grid>
  );
};

export default SelectModel;
