�
��
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring �
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.3.02v2.3.0-rc2-23-gb36436b0878��
�
conv3d_ind_0_0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameconv3d_ind_0_0/kernel
�
)conv3d_ind_0_0/kernel/Read/ReadVariableOpReadVariableOpconv3d_ind_0_0/kernel**
_output_shapes
:*
dtype0
~
conv3d_ind_0_0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameconv3d_ind_0_0/bias
w
'conv3d_ind_0_0/bias/Read/ReadVariableOpReadVariableOpconv3d_ind_0_0/bias*
_output_shapes
:*
dtype0
�
conv3d_ind_0_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameconv3d_ind_0_1/kernel
�
)conv3d_ind_0_1/kernel/Read/ReadVariableOpReadVariableOpconv3d_ind_0_1/kernel**
_output_shapes
:*
dtype0
~
conv3d_ind_0_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameconv3d_ind_0_1/bias
w
'conv3d_ind_0_1/bias/Read/ReadVariableOpReadVariableOpconv3d_ind_0_1/bias*
_output_shapes
:*
dtype0
�
prelu_ind_0_0/alphaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameprelu_ind_0_0/alpha
�
'prelu_ind_0_0/alpha/Read/ReadVariableOpReadVariableOpprelu_ind_0_0/alpha*&
_output_shapes
:*
dtype0
�
prelu_ind_0_1/alphaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameprelu_ind_0_1/alpha
�
'prelu_ind_0_1/alpha/Read/ReadVariableOpReadVariableOpprelu_ind_0_1/alpha*&
_output_shapes
:*
dtype0
�
conv3d_0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv3d_0/kernel

#conv3d_0/kernel/Read/ReadVariableOpReadVariableOpconv3d_0/kernel**
_output_shapes
:*
dtype0
r
conv3d_0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv3d_0/bias
k
!conv3d_0/bias/Read/ReadVariableOpReadVariableOpconv3d_0/bias*
_output_shapes
:*
dtype0
~
prelu_0/alphaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameprelu_0/alpha
w
!prelu_0/alpha/Read/ReadVariableOpReadVariableOpprelu_0/alpha*&
_output_shapes
:*
dtype0
�
conv3d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv3d_1/kernel

#conv3d_1/kernel/Read/ReadVariableOpReadVariableOpconv3d_1/kernel**
_output_shapes
:*
dtype0
r
conv3d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv3d_1/bias
k
!conv3d_1/bias/Read/ReadVariableOpReadVariableOpconv3d_1/bias*
_output_shapes
:*
dtype0
~
prelu_1/alphaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameprelu_1/alpha
w
!prelu_1/alpha/Read/ReadVariableOpReadVariableOpprelu_1/alpha*&
_output_shapes
:*
dtype0
�
conv3d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv3d_2/kernel

#conv3d_2/kernel/Read/ReadVariableOpReadVariableOpconv3d_2/kernel**
_output_shapes
:*
dtype0
r
conv3d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv3d_2/bias
k
!conv3d_2/bias/Read/ReadVariableOpReadVariableOpconv3d_2/bias*
_output_shapes
:*
dtype0
~
prelu_2/alphaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameprelu_2/alpha
w
!prelu_2/alpha/Read/ReadVariableOpReadVariableOpprelu_2/alpha*&
_output_shapes
:*
dtype0
�
conv3d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv3d_3/kernel

#conv3d_3/kernel/Read/ReadVariableOpReadVariableOpconv3d_3/kernel**
_output_shapes
:*
dtype0
r
conv3d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv3d_3/bias
k
!conv3d_3/bias/Read/ReadVariableOpReadVariableOpconv3d_3/bias*
_output_shapes
:*
dtype0
~
prelu_3/alphaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameprelu_3/alpha
w
!prelu_3/alpha/Read/ReadVariableOpReadVariableOpprelu_3/alpha*&
_output_shapes
:*
dtype0
�
conv3d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv3d_4/kernel

#conv3d_4/kernel/Read/ReadVariableOpReadVariableOpconv3d_4/kernel**
_output_shapes
:*
dtype0
r
conv3d_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv3d_4/bias
k
!conv3d_4/bias/Read/ReadVariableOpReadVariableOpconv3d_4/bias*
_output_shapes
:*
dtype0
~
prelu_4/alphaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameprelu_4/alpha
w
!prelu_4/alpha/Read/ReadVariableOpReadVariableOpprelu_4/alpha*&
_output_shapes
:*
dtype0
�
conv3d_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv3d_5/kernel

#conv3d_5/kernel/Read/ReadVariableOpReadVariableOpconv3d_5/kernel**
_output_shapes
:*
dtype0
r
conv3d_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv3d_5/bias
k
!conv3d_5/bias/Read/ReadVariableOpReadVariableOpconv3d_5/bias*
_output_shapes
:*
dtype0
~
prelu_5/alphaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameprelu_5/alpha
w
!prelu_5/alpha/Read/ReadVariableOpReadVariableOpprelu_5/alpha*&
_output_shapes
:*
dtype0
�
conv3d_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv3d_6/kernel

#conv3d_6/kernel/Read/ReadVariableOpReadVariableOpconv3d_6/kernel**
_output_shapes
:*
dtype0
r
conv3d_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv3d_6/bias
k
!conv3d_6/bias/Read/ReadVariableOpReadVariableOpconv3d_6/bias*
_output_shapes
:*
dtype0
~
prelu_6/alphaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameprelu_6/alpha
w
!prelu_6/alpha/Read/ReadVariableOpReadVariableOpprelu_6/alpha*&
_output_shapes
:*
dtype0
�
conv_111/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv_111/kernel

#conv_111/kernel/Read/ReadVariableOpReadVariableOpconv_111/kernel**
_output_shapes
:*
dtype0
r
conv_111/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv_111/bias
k
!conv_111/bias/Read/ReadVariableOpReadVariableOpconv_111/bias*
_output_shapes
:*
dtype0

NoOpNoOp
�X
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�X
value�XB�X B�X
�
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer_with_weights-4
layer-7
	layer_with_weights-5
	layer-8

layer_with_weights-6

layer-9
layer_with_weights-7
layer-10
layer_with_weights-8
layer-11
layer_with_weights-9
layer-12
layer_with_weights-10
layer-13
layer_with_weights-11
layer-14
layer_with_weights-12
layer-15
layer_with_weights-13
layer-16
layer_with_weights-14
layer-17
layer_with_weights-15
layer-18
layer_with_weights-16
layer-19
layer_with_weights-17
layer-20
layer_with_weights-18
layer-21
layer-22
trainable_variables
regularization_losses
	variables
	keras_api

signatures
 
 
h

kernel
bias
trainable_variables
 regularization_losses
!	variables
"	keras_api
h

#kernel
$bias
%trainable_variables
&regularization_losses
'	variables
(	keras_api
n
)shared_axes
	*alpha
+trainable_variables
,regularization_losses
-	variables
.	keras_api
n
/shared_axes
	0alpha
1trainable_variables
2regularization_losses
3	variables
4	keras_api
R
5trainable_variables
6regularization_losses
7	variables
8	keras_api
h

9kernel
:bias
;trainable_variables
<regularization_losses
=	variables
>	keras_api
n
?shared_axes
	@alpha
Atrainable_variables
Bregularization_losses
C	variables
D	keras_api
h

Ekernel
Fbias
Gtrainable_variables
Hregularization_losses
I	variables
J	keras_api
n
Kshared_axes
	Lalpha
Mtrainable_variables
Nregularization_losses
O	variables
P	keras_api
h

Qkernel
Rbias
Strainable_variables
Tregularization_losses
U	variables
V	keras_api
n
Wshared_axes
	Xalpha
Ytrainable_variables
Zregularization_losses
[	variables
\	keras_api
h

]kernel
^bias
_trainable_variables
`regularization_losses
a	variables
b	keras_api
n
cshared_axes
	dalpha
etrainable_variables
fregularization_losses
g	variables
h	keras_api
h

ikernel
jbias
ktrainable_variables
lregularization_losses
m	variables
n	keras_api
n
oshared_axes
	palpha
qtrainable_variables
rregularization_losses
s	variables
t	keras_api
h

ukernel
vbias
wtrainable_variables
xregularization_losses
y	variables
z	keras_api
o
{shared_axes
	|alpha
}trainable_variables
~regularization_losses
	variables
�	keras_api
n
�kernel
	�bias
�trainable_variables
�regularization_losses
�	variables
�	keras_api
t
�shared_axes

�alpha
�trainable_variables
�regularization_losses
�	variables
�	keras_api
n
�kernel
	�bias
�trainable_variables
�regularization_losses
�	variables
�	keras_api
V
�trainable_variables
�regularization_losses
�	variables
�	keras_api
�
0
1
#2
$3
*4
05
96
:7
@8
E9
F10
L11
Q12
R13
X14
]15
^16
d17
i18
j19
p20
u21
v22
|23
�24
�25
�26
�27
�28
 
�
0
1
#2
$3
*4
05
96
:7
@8
E9
F10
L11
Q12
R13
X14
]15
^16
d17
i18
j19
p20
u21
v22
|23
�24
�25
�26
�27
�28
�
�layer_metrics
�metrics
trainable_variables
�non_trainable_variables
regularization_losses
 �layer_regularization_losses
�layers
	variables
 
a_
VARIABLE_VALUEconv3d_ind_0_0/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEconv3d_ind_0_0/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
�
�layer_metrics
�metrics
trainable_variables
�non_trainable_variables
 regularization_losses
 �layer_regularization_losses
�layers
!	variables
a_
VARIABLE_VALUEconv3d_ind_0_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEconv3d_ind_0_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

#0
$1
 

#0
$1
�
�layer_metrics
�metrics
%trainable_variables
�non_trainable_variables
&regularization_losses
 �layer_regularization_losses
�layers
'	variables
 
^\
VARIABLE_VALUEprelu_ind_0_0/alpha5layer_with_weights-2/alpha/.ATTRIBUTES/VARIABLE_VALUE

*0
 

*0
�
�layer_metrics
�metrics
+trainable_variables
�non_trainable_variables
,regularization_losses
 �layer_regularization_losses
�layers
-	variables
 
^\
VARIABLE_VALUEprelu_ind_0_1/alpha5layer_with_weights-3/alpha/.ATTRIBUTES/VARIABLE_VALUE

00
 

00
�
�layer_metrics
�metrics
1trainable_variables
�non_trainable_variables
2regularization_losses
 �layer_regularization_losses
�layers
3	variables
 
 
 
�
�layer_metrics
�metrics
5trainable_variables
�non_trainable_variables
6regularization_losses
 �layer_regularization_losses
�layers
7	variables
[Y
VARIABLE_VALUEconv3d_0/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv3d_0/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

90
:1
 

90
:1
�
�layer_metrics
�metrics
;trainable_variables
�non_trainable_variables
<regularization_losses
 �layer_regularization_losses
�layers
=	variables
 
XV
VARIABLE_VALUEprelu_0/alpha5layer_with_weights-5/alpha/.ATTRIBUTES/VARIABLE_VALUE

@0
 

@0
�
�layer_metrics
�metrics
Atrainable_variables
�non_trainable_variables
Bregularization_losses
 �layer_regularization_losses
�layers
C	variables
[Y
VARIABLE_VALUEconv3d_1/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv3d_1/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

E0
F1
 

E0
F1
�
�layer_metrics
�metrics
Gtrainable_variables
�non_trainable_variables
Hregularization_losses
 �layer_regularization_losses
�layers
I	variables
 
XV
VARIABLE_VALUEprelu_1/alpha5layer_with_weights-7/alpha/.ATTRIBUTES/VARIABLE_VALUE

L0
 

L0
�
�layer_metrics
�metrics
Mtrainable_variables
�non_trainable_variables
Nregularization_losses
 �layer_regularization_losses
�layers
O	variables
[Y
VARIABLE_VALUEconv3d_2/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv3d_2/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE

Q0
R1
 

Q0
R1
�
�layer_metrics
�metrics
Strainable_variables
�non_trainable_variables
Tregularization_losses
 �layer_regularization_losses
�layers
U	variables
 
XV
VARIABLE_VALUEprelu_2/alpha5layer_with_weights-9/alpha/.ATTRIBUTES/VARIABLE_VALUE

X0
 

X0
�
�layer_metrics
�metrics
Ytrainable_variables
�non_trainable_variables
Zregularization_losses
 �layer_regularization_losses
�layers
[	variables
\Z
VARIABLE_VALUEconv3d_3/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv3d_3/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE

]0
^1
 

]0
^1
�
�layer_metrics
�metrics
_trainable_variables
�non_trainable_variables
`regularization_losses
 �layer_regularization_losses
�layers
a	variables
 
YW
VARIABLE_VALUEprelu_3/alpha6layer_with_weights-11/alpha/.ATTRIBUTES/VARIABLE_VALUE

d0
 

d0
�
�layer_metrics
�metrics
etrainable_variables
�non_trainable_variables
fregularization_losses
 �layer_regularization_losses
�layers
g	variables
\Z
VARIABLE_VALUEconv3d_4/kernel7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv3d_4/bias5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUE

i0
j1
 

i0
j1
�
�layer_metrics
�metrics
ktrainable_variables
�non_trainable_variables
lregularization_losses
 �layer_regularization_losses
�layers
m	variables
 
YW
VARIABLE_VALUEprelu_4/alpha6layer_with_weights-13/alpha/.ATTRIBUTES/VARIABLE_VALUE

p0
 

p0
�
�layer_metrics
�metrics
qtrainable_variables
�non_trainable_variables
rregularization_losses
 �layer_regularization_losses
�layers
s	variables
\Z
VARIABLE_VALUEconv3d_5/kernel7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv3d_5/bias5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUE

u0
v1
 

u0
v1
�
�layer_metrics
�metrics
wtrainable_variables
�non_trainable_variables
xregularization_losses
 �layer_regularization_losses
�layers
y	variables
 
YW
VARIABLE_VALUEprelu_5/alpha6layer_with_weights-15/alpha/.ATTRIBUTES/VARIABLE_VALUE

|0
 

|0
�
�layer_metrics
�metrics
}trainable_variables
�non_trainable_variables
~regularization_losses
 �layer_regularization_losses
�layers
	variables
\Z
VARIABLE_VALUEconv3d_6/kernel7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv3d_6/bias5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUE

�0
�1
 

�0
�1
�
�layer_metrics
�metrics
�trainable_variables
�non_trainable_variables
�regularization_losses
 �layer_regularization_losses
�layers
�	variables
 
YW
VARIABLE_VALUEprelu_6/alpha6layer_with_weights-17/alpha/.ATTRIBUTES/VARIABLE_VALUE

�0
 

�0
�
�layer_metrics
�metrics
�trainable_variables
�non_trainable_variables
�regularization_losses
 �layer_regularization_losses
�layers
�	variables
\Z
VARIABLE_VALUEconv_111/kernel7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv_111/bias5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUE

�0
�1
 

�0
�1
�
�layer_metrics
�metrics
�trainable_variables
�non_trainable_variables
�regularization_losses
 �layer_regularization_losses
�layers
�	variables
 
 
 
�
�layer_metrics
�metrics
�trainable_variables
�non_trainable_variables
�regularization_losses
 �layer_regularization_losses
�layers
�	variables
 
 
 
 
�
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
�
serving_default_input_0Placeholder*N
_output_shapes<
::8������������������������������������*
dtype0*C
shape::8������������������������������������
�
serving_default_input_1Placeholder*N
_output_shapes<
::8������������������������������������*
dtype0*C
shape::8������������������������������������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_0serving_default_input_1conv3d_ind_0_1/kernelconv3d_ind_0_1/biasconv3d_ind_0_0/kernelconv3d_ind_0_0/biasprelu_ind_0_0/alphaprelu_ind_0_1/alphaconv3d_0/kernelconv3d_0/biasprelu_0/alphaconv3d_1/kernelconv3d_1/biasprelu_1/alphaconv3d_2/kernelconv3d_2/biasprelu_2/alphaconv3d_3/kernelconv3d_3/biasprelu_3/alphaconv3d_4/kernelconv3d_4/biasprelu_4/alphaconv3d_5/kernelconv3d_5/biasprelu_5/alphaconv3d_6/kernelconv3d_6/biasprelu_6/alphaconv_111/kernelconv_111/bias**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8������������������������������������*?
_read_only_resource_inputs!
	
*-
config_proto

CPU

GPU 2J 8� *+
f&R$
"__inference_signature_wrapper_5076
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename)conv3d_ind_0_0/kernel/Read/ReadVariableOp'conv3d_ind_0_0/bias/Read/ReadVariableOp)conv3d_ind_0_1/kernel/Read/ReadVariableOp'conv3d_ind_0_1/bias/Read/ReadVariableOp'prelu_ind_0_0/alpha/Read/ReadVariableOp'prelu_ind_0_1/alpha/Read/ReadVariableOp#conv3d_0/kernel/Read/ReadVariableOp!conv3d_0/bias/Read/ReadVariableOp!prelu_0/alpha/Read/ReadVariableOp#conv3d_1/kernel/Read/ReadVariableOp!conv3d_1/bias/Read/ReadVariableOp!prelu_1/alpha/Read/ReadVariableOp#conv3d_2/kernel/Read/ReadVariableOp!conv3d_2/bias/Read/ReadVariableOp!prelu_2/alpha/Read/ReadVariableOp#conv3d_3/kernel/Read/ReadVariableOp!conv3d_3/bias/Read/ReadVariableOp!prelu_3/alpha/Read/ReadVariableOp#conv3d_4/kernel/Read/ReadVariableOp!conv3d_4/bias/Read/ReadVariableOp!prelu_4/alpha/Read/ReadVariableOp#conv3d_5/kernel/Read/ReadVariableOp!conv3d_5/bias/Read/ReadVariableOp!prelu_5/alpha/Read/ReadVariableOp#conv3d_6/kernel/Read/ReadVariableOp!conv3d_6/bias/Read/ReadVariableOp!prelu_6/alpha/Read/ReadVariableOp#conv_111/kernel/Read/ReadVariableOp!conv_111/bias/Read/ReadVariableOpConst**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *&
f!R
__inference__traced_save_5810
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv3d_ind_0_0/kernelconv3d_ind_0_0/biasconv3d_ind_0_1/kernelconv3d_ind_0_1/biasprelu_ind_0_0/alphaprelu_ind_0_1/alphaconv3d_0/kernelconv3d_0/biasprelu_0/alphaconv3d_1/kernelconv3d_1/biasprelu_1/alphaconv3d_2/kernelconv3d_2/biasprelu_2/alphaconv3d_3/kernelconv3d_3/biasprelu_3/alphaconv3d_4/kernelconv3d_4/biasprelu_4/alphaconv3d_5/kernelconv3d_5/biasprelu_5/alphaconv3d_6/kernelconv3d_6/biasprelu_6/alphaconv_111/kernelconv_111/bias*)
Tin"
 2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *)
f$R"
 __inference__traced_restore_5907�
�
l
&__inference_prelu_0_layer_call_fn_4179

inputs
unknown
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8������������������������������������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_prelu_0_layer_call_and_return_conditional_losses_41712
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*N
_output_shapes<
::8������������������������������������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:8������������������������������������:22
StatefulPartitionedCallStatefulPartitionedCall:v r
N
_output_shapes<
::8������������������������������������
 
_user_specified_nameinputs
�
�
,__inference_functional_15_layer_call_fn_5010
input_0
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_0input_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8������������������������������������*?
_read_only_resource_inputs!
	
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_functional_15_layer_call_and_return_conditional_losses_49492
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*N
_output_shapes<
::8������������������������������������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:8������������������������������������:8������������������������������������:::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:w s
N
_output_shapes<
::8������������������������������������
!
_user_specified_name	input_0:ws
N
_output_shapes<
::8������������������������������������
!
_user_specified_name	input_1
�l
�

G__inference_functional_15_layer_call_and_return_conditional_losses_4801

inputs
inputs_1
conv3d_ind_0_1_4721
conv3d_ind_0_1_4723
conv3d_ind_0_0_4726
conv3d_ind_0_0_4728
prelu_ind_0_0_4731
prelu_ind_0_1_4734
conv3d_0_4738
conv3d_0_4740
prelu_0_4743
conv3d_1_4746
conv3d_1_4748
prelu_1_4751
conv3d_2_4754
conv3d_2_4756
prelu_2_4759
conv3d_3_4762
conv3d_3_4764
prelu_3_4767
conv3d_4_4770
conv3d_4_4772
prelu_4_4775
conv3d_5_4778
conv3d_5_4780
prelu_5_4783
conv3d_6_4786
conv3d_6_4788
prelu_6_4791
conv_111_4794
conv_111_4796
identity�� conv3d_0/StatefulPartitionedCall� conv3d_1/StatefulPartitionedCall� conv3d_2/StatefulPartitionedCall� conv3d_3/StatefulPartitionedCall� conv3d_4/StatefulPartitionedCall� conv3d_5/StatefulPartitionedCall� conv3d_6/StatefulPartitionedCall�&conv3d_ind_0_0/StatefulPartitionedCall�&conv3d_ind_0_1/StatefulPartitionedCall� conv_111/StatefulPartitionedCall�prelu_0/StatefulPartitionedCall�prelu_1/StatefulPartitionedCall�prelu_2/StatefulPartitionedCall�prelu_3/StatefulPartitionedCall�prelu_4/StatefulPartitionedCall�prelu_5/StatefulPartitionedCall�prelu_6/StatefulPartitionedCall�%prelu_ind_0_0/StatefulPartitionedCall�%prelu_ind_0_1/StatefulPartitionedCall�
&conv3d_ind_0_1/StatefulPartitionedCallStatefulPartitionedCallinputs_1conv3d_ind_0_1_4721conv3d_ind_0_1_4723*
Tin
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8������������������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_conv3d_ind_0_1_layer_call_and_return_conditional_losses_43202(
&conv3d_ind_0_1/StatefulPartitionedCall�
&conv3d_ind_0_0/StatefulPartitionedCallStatefulPartitionedCallinputsconv3d_ind_0_0_4726conv3d_ind_0_0_4728*
Tin
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8������������������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_conv3d_ind_0_0_layer_call_and_return_conditional_losses_43462(
&conv3d_ind_0_0/StatefulPartitionedCall�
%prelu_ind_0_0/StatefulPartitionedCallStatefulPartitionedCall/conv3d_ind_0_0/StatefulPartitionedCall:output:0prelu_ind_0_0_4731*
Tin
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8������������������������������������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_prelu_ind_0_0_layer_call_and_return_conditional_losses_41292'
%prelu_ind_0_0/StatefulPartitionedCall�
%prelu_ind_0_1/StatefulPartitionedCallStatefulPartitionedCall/conv3d_ind_0_1/StatefulPartitionedCall:output:0prelu_ind_0_1_4734*
Tin
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8������������������������������������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_prelu_ind_0_1_layer_call_and_return_conditional_losses_41502'
%prelu_ind_0_1/StatefulPartitionedCall�
concat_0/PartitionedCallPartitionedCall.prelu_ind_0_0/StatefulPartitionedCall:output:0.prelu_ind_0_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8������������������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_concat_0_layer_call_and_return_conditional_losses_43752
concat_0/PartitionedCall�
 conv3d_0/StatefulPartitionedCallStatefulPartitionedCall!concat_0/PartitionedCall:output:0conv3d_0_4738conv3d_0_4740*
Tin
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8������������������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_conv3d_0_layer_call_and_return_conditional_losses_43942"
 conv3d_0/StatefulPartitionedCall�
prelu_0/StatefulPartitionedCallStatefulPartitionedCall)conv3d_0/StatefulPartitionedCall:output:0prelu_0_4743*
Tin
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8������������������������������������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_prelu_0_layer_call_and_return_conditional_losses_41712!
prelu_0/StatefulPartitionedCall�
 conv3d_1/StatefulPartitionedCallStatefulPartitionedCall(prelu_0/StatefulPartitionedCall:output:0conv3d_1_4746conv3d_1_4748*
Tin
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8������������������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_conv3d_1_layer_call_and_return_conditional_losses_44232"
 conv3d_1/StatefulPartitionedCall�
prelu_1/StatefulPartitionedCallStatefulPartitionedCall)conv3d_1/StatefulPartitionedCall:output:0prelu_1_4751*
Tin
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8������������������������������������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_prelu_1_layer_call_and_return_conditional_losses_41922!
prelu_1/StatefulPartitionedCall�
 conv3d_2/StatefulPartitionedCallStatefulPartitionedCall(prelu_1/StatefulPartitionedCall:output:0conv3d_2_4754conv3d_2_4756*
Tin
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8������������������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_conv3d_2_layer_call_and_return_conditional_losses_44522"
 conv3d_2/StatefulPartitionedCall�
prelu_2/StatefulPartitionedCallStatefulPartitionedCall)conv3d_2/StatefulPartitionedCall:output:0prelu_2_4759*
Tin
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8������������������������������������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_prelu_2_layer_call_and_return_conditional_losses_42132!
prelu_2/StatefulPartitionedCall�
 conv3d_3/StatefulPartitionedCallStatefulPartitionedCall(prelu_2/StatefulPartitionedCall:output:0conv3d_3_4762conv3d_3_4764*
Tin
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8������������������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_conv3d_3_layer_call_and_return_conditional_losses_44812"
 conv3d_3/StatefulPartitionedCall�
prelu_3/StatefulPartitionedCallStatefulPartitionedCall)conv3d_3/StatefulPartitionedCall:output:0prelu_3_4767*
Tin
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8������������������������������������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_prelu_3_layer_call_and_return_conditional_losses_42342!
prelu_3/StatefulPartitionedCall�
 conv3d_4/StatefulPartitionedCallStatefulPartitionedCall(prelu_3/StatefulPartitionedCall:output:0conv3d_4_4770conv3d_4_4772*
Tin
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8������������������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_conv3d_4_layer_call_and_return_conditional_losses_45102"
 conv3d_4/StatefulPartitionedCall�
prelu_4/StatefulPartitionedCallStatefulPartitionedCall)conv3d_4/StatefulPartitionedCall:output:0prelu_4_4775*
Tin
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8������������������������������������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_prelu_4_layer_call_and_return_conditional_losses_42552!
prelu_4/StatefulPartitionedCall�
 conv3d_5/StatefulPartitionedCallStatefulPartitionedCall(prelu_4/StatefulPartitionedCall:output:0conv3d_5_4778conv3d_5_4780*
Tin
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8������������������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_conv3d_5_layer_call_and_return_conditional_losses_45392"
 conv3d_5/StatefulPartitionedCall�
prelu_5/StatefulPartitionedCallStatefulPartitionedCall)conv3d_5/StatefulPartitionedCall:output:0prelu_5_4783*
Tin
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8������������������������������������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_prelu_5_layer_call_and_return_conditional_losses_42762!
prelu_5/StatefulPartitionedCall�
 conv3d_6/StatefulPartitionedCallStatefulPartitionedCall(prelu_5/StatefulPartitionedCall:output:0conv3d_6_4786conv3d_6_4788*
Tin
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8������������������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_conv3d_6_layer_call_and_return_conditional_losses_45682"
 conv3d_6/StatefulPartitionedCall�
prelu_6/StatefulPartitionedCallStatefulPartitionedCall)conv3d_6/StatefulPartitionedCall:output:0prelu_6_4791*
Tin
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8������������������������������������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_prelu_6_layer_call_and_return_conditional_losses_42972!
prelu_6/StatefulPartitionedCall�
 conv_111/StatefulPartitionedCallStatefulPartitionedCall(prelu_6/StatefulPartitionedCall:output:0conv_111_4794conv_111_4796*
Tin
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8������������������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_conv_111_layer_call_and_return_conditional_losses_45972"
 conv_111/StatefulPartitionedCall�
add_0/PartitionedCallPartitionedCall)conv_111/StatefulPartitionedCall:output:0inputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8������������������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_add_0_layer_call_and_return_conditional_losses_46192
add_0/PartitionedCall�
IdentityIdentityadd_0/PartitionedCall:output:0!^conv3d_0/StatefulPartitionedCall!^conv3d_1/StatefulPartitionedCall!^conv3d_2/StatefulPartitionedCall!^conv3d_3/StatefulPartitionedCall!^conv3d_4/StatefulPartitionedCall!^conv3d_5/StatefulPartitionedCall!^conv3d_6/StatefulPartitionedCall'^conv3d_ind_0_0/StatefulPartitionedCall'^conv3d_ind_0_1/StatefulPartitionedCall!^conv_111/StatefulPartitionedCall ^prelu_0/StatefulPartitionedCall ^prelu_1/StatefulPartitionedCall ^prelu_2/StatefulPartitionedCall ^prelu_3/StatefulPartitionedCall ^prelu_4/StatefulPartitionedCall ^prelu_5/StatefulPartitionedCall ^prelu_6/StatefulPartitionedCall&^prelu_ind_0_0/StatefulPartitionedCall&^prelu_ind_0_1/StatefulPartitionedCall*
T0*N
_output_shapes<
::8������������������������������������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:8������������������������������������:8������������������������������������:::::::::::::::::::::::::::::2D
 conv3d_0/StatefulPartitionedCall conv3d_0/StatefulPartitionedCall2D
 conv3d_1/StatefulPartitionedCall conv3d_1/StatefulPartitionedCall2D
 conv3d_2/StatefulPartitionedCall conv3d_2/StatefulPartitionedCall2D
 conv3d_3/StatefulPartitionedCall conv3d_3/StatefulPartitionedCall2D
 conv3d_4/StatefulPartitionedCall conv3d_4/StatefulPartitionedCall2D
 conv3d_5/StatefulPartitionedCall conv3d_5/StatefulPartitionedCall2D
 conv3d_6/StatefulPartitionedCall conv3d_6/StatefulPartitionedCall2P
&conv3d_ind_0_0/StatefulPartitionedCall&conv3d_ind_0_0/StatefulPartitionedCall2P
&conv3d_ind_0_1/StatefulPartitionedCall&conv3d_ind_0_1/StatefulPartitionedCall2D
 conv_111/StatefulPartitionedCall conv_111/StatefulPartitionedCall2B
prelu_0/StatefulPartitionedCallprelu_0/StatefulPartitionedCall2B
prelu_1/StatefulPartitionedCallprelu_1/StatefulPartitionedCall2B
prelu_2/StatefulPartitionedCallprelu_2/StatefulPartitionedCall2B
prelu_3/StatefulPartitionedCallprelu_3/StatefulPartitionedCall2B
prelu_4/StatefulPartitionedCallprelu_4/StatefulPartitionedCall2B
prelu_5/StatefulPartitionedCallprelu_5/StatefulPartitionedCall2B
prelu_6/StatefulPartitionedCallprelu_6/StatefulPartitionedCall2N
%prelu_ind_0_0/StatefulPartitionedCall%prelu_ind_0_0/StatefulPartitionedCall2N
%prelu_ind_0_1/StatefulPartitionedCall%prelu_ind_0_1/StatefulPartitionedCall:v r
N
_output_shapes<
::8������������������������������������
 
_user_specified_nameinputs:vr
N
_output_shapes<
::8������������������������������������
 
_user_specified_nameinputs
�	
�
H__inference_conv3d_ind_0_1_layer_call_and_return_conditional_losses_5513

inputs"
conv3d_readvariableop_resource#
biasadd_readvariableop_resource
identity��
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype02
Conv3D/ReadVariableOp�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8������������������������������������*
paddingSAME*
strides	
2
Conv3D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8������������������������������������2	
BiasAdd�
IdentityIdentityBiasAdd:output:0*
T0*N
_output_shapes<
::8������������������������������������2

Identity"
identityIdentity:output:0*U
_input_shapesD
B:8������������������������������������:::v r
N
_output_shapes<
::8������������������������������������
 
_user_specified_nameinputs
�	
�
B__inference_conv3d_5_layer_call_and_return_conditional_losses_5640

inputs"
conv3d_readvariableop_resource#
biasadd_readvariableop_resource
identity��
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype02
Conv3D/ReadVariableOp�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8������������������������������������*
paddingSAME*
strides	
2
Conv3D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8������������������������������������2	
BiasAdd�
IdentityIdentityBiasAdd:output:0*
T0*N
_output_shapes<
::8������������������������������������2

Identity"
identityIdentity:output:0*U
_input_shapesD
B:8������������������������������������:::v r
N
_output_shapes<
::8������������������������������������
 
_user_specified_nameinputs
�
|
'__inference_conv3d_0_layer_call_fn_5554

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8������������������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_conv3d_0_layer_call_and_return_conditional_losses_43942
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*N
_output_shapes<
::8������������������������������������2

Identity"
identityIdentity:output:0*U
_input_shapesD
B:8������������������������������������::22
StatefulPartitionedCallStatefulPartitionedCall:v r
N
_output_shapes<
::8������������������������������������
 
_user_specified_nameinputs
�
|
'__inference_conv_111_layer_call_fn_5687

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8������������������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_conv_111_layer_call_and_return_conditional_losses_45972
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*N
_output_shapes<
::8������������������������������������2

Identity"
identityIdentity:output:0*U
_input_shapesD
B:8������������������������������������::22
StatefulPartitionedCallStatefulPartitionedCall:v r
N
_output_shapes<
::8������������������������������������
 
_user_specified_nameinputs
�

�
G__inference_prelu_ind_0_1_layer_call_and_return_conditional_losses_4150

inputs
readvariableop_resource
identity�u
ReluReluinputs*
T0*N
_output_shapes<
::8������������������������������������2
Relu�
ReadVariableOpReadVariableOpreadvariableop_resource*&
_output_shapes
:*
dtype02
ReadVariableOpZ
NegNegReadVariableOp:value:0*
T0*&
_output_shapes
:2
Negv
Neg_1Neginputs*
T0*N
_output_shapes<
::8������������������������������������2
Neg_1|
Relu_1Relu	Neg_1:y:0*
T0*N
_output_shapes<
::8������������������������������������2
Relu_1�
mulMulNeg:y:0Relu_1:activations:0*
T0*N
_output_shapes<
::8������������������������������������2
mul�
addAddV2Relu:activations:0mul:z:0*
T0*N
_output_shapes<
::8������������������������������������2
add�
IdentityIdentityadd:z:0*
T0*N
_output_shapes<
::8������������������������������������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:8������������������������������������::v r
N
_output_shapes<
::8������������������������������������
 
_user_specified_nameinputs
�	
�
B__inference_conv3d_3_layer_call_and_return_conditional_losses_5602

inputs"
conv3d_readvariableop_resource#
biasadd_readvariableop_resource
identity��
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype02
Conv3D/ReadVariableOp�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8������������������������������������*
paddingSAME*
strides	
2
Conv3D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8������������������������������������2	
BiasAdd�
IdentityIdentityBiasAdd:output:0*
T0*N
_output_shapes<
::8������������������������������������2

Identity"
identityIdentity:output:0*U
_input_shapesD
B:8������������������������������������:::v r
N
_output_shapes<
::8������������������������������������
 
_user_specified_nameinputs
��
�

G__inference_functional_15_layer_call_and_return_conditional_losses_5216
inputs_0
inputs_11
-conv3d_ind_0_1_conv3d_readvariableop_resource2
.conv3d_ind_0_1_biasadd_readvariableop_resource1
-conv3d_ind_0_0_conv3d_readvariableop_resource2
.conv3d_ind_0_0_biasadd_readvariableop_resource)
%prelu_ind_0_0_readvariableop_resource)
%prelu_ind_0_1_readvariableop_resource+
'conv3d_0_conv3d_readvariableop_resource,
(conv3d_0_biasadd_readvariableop_resource#
prelu_0_readvariableop_resource+
'conv3d_1_conv3d_readvariableop_resource,
(conv3d_1_biasadd_readvariableop_resource#
prelu_1_readvariableop_resource+
'conv3d_2_conv3d_readvariableop_resource,
(conv3d_2_biasadd_readvariableop_resource#
prelu_2_readvariableop_resource+
'conv3d_3_conv3d_readvariableop_resource,
(conv3d_3_biasadd_readvariableop_resource#
prelu_3_readvariableop_resource+
'conv3d_4_conv3d_readvariableop_resource,
(conv3d_4_biasadd_readvariableop_resource#
prelu_4_readvariableop_resource+
'conv3d_5_conv3d_readvariableop_resource,
(conv3d_5_biasadd_readvariableop_resource#
prelu_5_readvariableop_resource+
'conv3d_6_conv3d_readvariableop_resource,
(conv3d_6_biasadd_readvariableop_resource#
prelu_6_readvariableop_resource+
'conv_111_conv3d_readvariableop_resource,
(conv_111_biasadd_readvariableop_resource
identity��
$conv3d_ind_0_1/Conv3D/ReadVariableOpReadVariableOp-conv3d_ind_0_1_conv3d_readvariableop_resource**
_output_shapes
:*
dtype02&
$conv3d_ind_0_1/Conv3D/ReadVariableOp�
conv3d_ind_0_1/Conv3DConv3Dinputs_1,conv3d_ind_0_1/Conv3D/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8������������������������������������*
paddingSAME*
strides	
2
conv3d_ind_0_1/Conv3D�
%conv3d_ind_0_1/BiasAdd/ReadVariableOpReadVariableOp.conv3d_ind_0_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%conv3d_ind_0_1/BiasAdd/ReadVariableOp�
conv3d_ind_0_1/BiasAddBiasAddconv3d_ind_0_1/Conv3D:output:0-conv3d_ind_0_1/BiasAdd/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8������������������������������������2
conv3d_ind_0_1/BiasAdd�
$conv3d_ind_0_0/Conv3D/ReadVariableOpReadVariableOp-conv3d_ind_0_0_conv3d_readvariableop_resource**
_output_shapes
:*
dtype02&
$conv3d_ind_0_0/Conv3D/ReadVariableOp�
conv3d_ind_0_0/Conv3DConv3Dinputs_0,conv3d_ind_0_0/Conv3D/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8������������������������������������*
paddingSAME*
strides	
2
conv3d_ind_0_0/Conv3D�
%conv3d_ind_0_0/BiasAdd/ReadVariableOpReadVariableOp.conv3d_ind_0_0_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%conv3d_ind_0_0/BiasAdd/ReadVariableOp�
conv3d_ind_0_0/BiasAddBiasAddconv3d_ind_0_0/Conv3D:output:0-conv3d_ind_0_0/BiasAdd/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8������������������������������������2
conv3d_ind_0_0/BiasAdd�
prelu_ind_0_0/ReluReluconv3d_ind_0_0/BiasAdd:output:0*
T0*N
_output_shapes<
::8������������������������������������2
prelu_ind_0_0/Relu�
prelu_ind_0_0/ReadVariableOpReadVariableOp%prelu_ind_0_0_readvariableop_resource*&
_output_shapes
:*
dtype02
prelu_ind_0_0/ReadVariableOp�
prelu_ind_0_0/NegNeg$prelu_ind_0_0/ReadVariableOp:value:0*
T0*&
_output_shapes
:2
prelu_ind_0_0/Neg�
prelu_ind_0_0/Neg_1Negconv3d_ind_0_0/BiasAdd:output:0*
T0*N
_output_shapes<
::8������������������������������������2
prelu_ind_0_0/Neg_1�
prelu_ind_0_0/Relu_1Reluprelu_ind_0_0/Neg_1:y:0*
T0*N
_output_shapes<
::8������������������������������������2
prelu_ind_0_0/Relu_1�
prelu_ind_0_0/mulMulprelu_ind_0_0/Neg:y:0"prelu_ind_0_0/Relu_1:activations:0*
T0*N
_output_shapes<
::8������������������������������������2
prelu_ind_0_0/mul�
prelu_ind_0_0/addAddV2 prelu_ind_0_0/Relu:activations:0prelu_ind_0_0/mul:z:0*
T0*N
_output_shapes<
::8������������������������������������2
prelu_ind_0_0/add�
prelu_ind_0_1/ReluReluconv3d_ind_0_1/BiasAdd:output:0*
T0*N
_output_shapes<
::8������������������������������������2
prelu_ind_0_1/Relu�
prelu_ind_0_1/ReadVariableOpReadVariableOp%prelu_ind_0_1_readvariableop_resource*&
_output_shapes
:*
dtype02
prelu_ind_0_1/ReadVariableOp�
prelu_ind_0_1/NegNeg$prelu_ind_0_1/ReadVariableOp:value:0*
T0*&
_output_shapes
:2
prelu_ind_0_1/Neg�
prelu_ind_0_1/Neg_1Negconv3d_ind_0_1/BiasAdd:output:0*
T0*N
_output_shapes<
::8������������������������������������2
prelu_ind_0_1/Neg_1�
prelu_ind_0_1/Relu_1Reluprelu_ind_0_1/Neg_1:y:0*
T0*N
_output_shapes<
::8������������������������������������2
prelu_ind_0_1/Relu_1�
prelu_ind_0_1/mulMulprelu_ind_0_1/Neg:y:0"prelu_ind_0_1/Relu_1:activations:0*
T0*N
_output_shapes<
::8������������������������������������2
prelu_ind_0_1/mul�
prelu_ind_0_1/addAddV2 prelu_ind_0_1/Relu:activations:0prelu_ind_0_1/mul:z:0*
T0*N
_output_shapes<
::8������������������������������������2
prelu_ind_0_1/addn
concat_0/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_0/concat/axis�
concat_0/concatConcatV2prelu_ind_0_0/add:z:0prelu_ind_0_1/add:z:0concat_0/concat/axis:output:0*
N*
T0*N
_output_shapes<
::8������������������������������������2
concat_0/concat�
conv3d_0/Conv3D/ReadVariableOpReadVariableOp'conv3d_0_conv3d_readvariableop_resource**
_output_shapes
:*
dtype02 
conv3d_0/Conv3D/ReadVariableOp�
conv3d_0/Conv3DConv3Dconcat_0/concat:output:0&conv3d_0/Conv3D/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8������������������������������������*
paddingSAME*
strides	
2
conv3d_0/Conv3D�
conv3d_0/BiasAdd/ReadVariableOpReadVariableOp(conv3d_0_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv3d_0/BiasAdd/ReadVariableOp�
conv3d_0/BiasAddBiasAddconv3d_0/Conv3D:output:0'conv3d_0/BiasAdd/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8������������������������������������2
conv3d_0/BiasAdd�
prelu_0/ReluReluconv3d_0/BiasAdd:output:0*
T0*N
_output_shapes<
::8������������������������������������2
prelu_0/Relu�
prelu_0/ReadVariableOpReadVariableOpprelu_0_readvariableop_resource*&
_output_shapes
:*
dtype02
prelu_0/ReadVariableOpr
prelu_0/NegNegprelu_0/ReadVariableOp:value:0*
T0*&
_output_shapes
:2
prelu_0/Neg�
prelu_0/Neg_1Negconv3d_0/BiasAdd:output:0*
T0*N
_output_shapes<
::8������������������������������������2
prelu_0/Neg_1�
prelu_0/Relu_1Reluprelu_0/Neg_1:y:0*
T0*N
_output_shapes<
::8������������������������������������2
prelu_0/Relu_1�
prelu_0/mulMulprelu_0/Neg:y:0prelu_0/Relu_1:activations:0*
T0*N
_output_shapes<
::8������������������������������������2
prelu_0/mul�
prelu_0/addAddV2prelu_0/Relu:activations:0prelu_0/mul:z:0*
T0*N
_output_shapes<
::8������������������������������������2
prelu_0/add�
conv3d_1/Conv3D/ReadVariableOpReadVariableOp'conv3d_1_conv3d_readvariableop_resource**
_output_shapes
:*
dtype02 
conv3d_1/Conv3D/ReadVariableOp�
conv3d_1/Conv3DConv3Dprelu_0/add:z:0&conv3d_1/Conv3D/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8������������������������������������*
paddingSAME*
strides	
2
conv3d_1/Conv3D�
conv3d_1/BiasAdd/ReadVariableOpReadVariableOp(conv3d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv3d_1/BiasAdd/ReadVariableOp�
conv3d_1/BiasAddBiasAddconv3d_1/Conv3D:output:0'conv3d_1/BiasAdd/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8������������������������������������2
conv3d_1/BiasAdd�
prelu_1/ReluReluconv3d_1/BiasAdd:output:0*
T0*N
_output_shapes<
::8������������������������������������2
prelu_1/Relu�
prelu_1/ReadVariableOpReadVariableOpprelu_1_readvariableop_resource*&
_output_shapes
:*
dtype02
prelu_1/ReadVariableOpr
prelu_1/NegNegprelu_1/ReadVariableOp:value:0*
T0*&
_output_shapes
:2
prelu_1/Neg�
prelu_1/Neg_1Negconv3d_1/BiasAdd:output:0*
T0*N
_output_shapes<
::8������������������������������������2
prelu_1/Neg_1�
prelu_1/Relu_1Reluprelu_1/Neg_1:y:0*
T0*N
_output_shapes<
::8������������������������������������2
prelu_1/Relu_1�
prelu_1/mulMulprelu_1/Neg:y:0prelu_1/Relu_1:activations:0*
T0*N
_output_shapes<
::8������������������������������������2
prelu_1/mul�
prelu_1/addAddV2prelu_1/Relu:activations:0prelu_1/mul:z:0*
T0*N
_output_shapes<
::8������������������������������������2
prelu_1/add�
conv3d_2/Conv3D/ReadVariableOpReadVariableOp'conv3d_2_conv3d_readvariableop_resource**
_output_shapes
:*
dtype02 
conv3d_2/Conv3D/ReadVariableOp�
conv3d_2/Conv3DConv3Dprelu_1/add:z:0&conv3d_2/Conv3D/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8������������������������������������*
paddingSAME*
strides	
2
conv3d_2/Conv3D�
conv3d_2/BiasAdd/ReadVariableOpReadVariableOp(conv3d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv3d_2/BiasAdd/ReadVariableOp�
conv3d_2/BiasAddBiasAddconv3d_2/Conv3D:output:0'conv3d_2/BiasAdd/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8������������������������������������2
conv3d_2/BiasAdd�
prelu_2/ReluReluconv3d_2/BiasAdd:output:0*
T0*N
_output_shapes<
::8������������������������������������2
prelu_2/Relu�
prelu_2/ReadVariableOpReadVariableOpprelu_2_readvariableop_resource*&
_output_shapes
:*
dtype02
prelu_2/ReadVariableOpr
prelu_2/NegNegprelu_2/ReadVariableOp:value:0*
T0*&
_output_shapes
:2
prelu_2/Neg�
prelu_2/Neg_1Negconv3d_2/BiasAdd:output:0*
T0*N
_output_shapes<
::8������������������������������������2
prelu_2/Neg_1�
prelu_2/Relu_1Reluprelu_2/Neg_1:y:0*
T0*N
_output_shapes<
::8������������������������������������2
prelu_2/Relu_1�
prelu_2/mulMulprelu_2/Neg:y:0prelu_2/Relu_1:activations:0*
T0*N
_output_shapes<
::8������������������������������������2
prelu_2/mul�
prelu_2/addAddV2prelu_2/Relu:activations:0prelu_2/mul:z:0*
T0*N
_output_shapes<
::8������������������������������������2
prelu_2/add�
conv3d_3/Conv3D/ReadVariableOpReadVariableOp'conv3d_3_conv3d_readvariableop_resource**
_output_shapes
:*
dtype02 
conv3d_3/Conv3D/ReadVariableOp�
conv3d_3/Conv3DConv3Dprelu_2/add:z:0&conv3d_3/Conv3D/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8������������������������������������*
paddingSAME*
strides	
2
conv3d_3/Conv3D�
conv3d_3/BiasAdd/ReadVariableOpReadVariableOp(conv3d_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv3d_3/BiasAdd/ReadVariableOp�
conv3d_3/BiasAddBiasAddconv3d_3/Conv3D:output:0'conv3d_3/BiasAdd/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8������������������������������������2
conv3d_3/BiasAdd�
prelu_3/ReluReluconv3d_3/BiasAdd:output:0*
T0*N
_output_shapes<
::8������������������������������������2
prelu_3/Relu�
prelu_3/ReadVariableOpReadVariableOpprelu_3_readvariableop_resource*&
_output_shapes
:*
dtype02
prelu_3/ReadVariableOpr
prelu_3/NegNegprelu_3/ReadVariableOp:value:0*
T0*&
_output_shapes
:2
prelu_3/Neg�
prelu_3/Neg_1Negconv3d_3/BiasAdd:output:0*
T0*N
_output_shapes<
::8������������������������������������2
prelu_3/Neg_1�
prelu_3/Relu_1Reluprelu_3/Neg_1:y:0*
T0*N
_output_shapes<
::8������������������������������������2
prelu_3/Relu_1�
prelu_3/mulMulprelu_3/Neg:y:0prelu_3/Relu_1:activations:0*
T0*N
_output_shapes<
::8������������������������������������2
prelu_3/mul�
prelu_3/addAddV2prelu_3/Relu:activations:0prelu_3/mul:z:0*
T0*N
_output_shapes<
::8������������������������������������2
prelu_3/add�
conv3d_4/Conv3D/ReadVariableOpReadVariableOp'conv3d_4_conv3d_readvariableop_resource**
_output_shapes
:*
dtype02 
conv3d_4/Conv3D/ReadVariableOp�
conv3d_4/Conv3DConv3Dprelu_3/add:z:0&conv3d_4/Conv3D/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8������������������������������������*
paddingSAME*
strides	
2
conv3d_4/Conv3D�
conv3d_4/BiasAdd/ReadVariableOpReadVariableOp(conv3d_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv3d_4/BiasAdd/ReadVariableOp�
conv3d_4/BiasAddBiasAddconv3d_4/Conv3D:output:0'conv3d_4/BiasAdd/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8������������������������������������2
conv3d_4/BiasAdd�
prelu_4/ReluReluconv3d_4/BiasAdd:output:0*
T0*N
_output_shapes<
::8������������������������������������2
prelu_4/Relu�
prelu_4/ReadVariableOpReadVariableOpprelu_4_readvariableop_resource*&
_output_shapes
:*
dtype02
prelu_4/ReadVariableOpr
prelu_4/NegNegprelu_4/ReadVariableOp:value:0*
T0*&
_output_shapes
:2
prelu_4/Neg�
prelu_4/Neg_1Negconv3d_4/BiasAdd:output:0*
T0*N
_output_shapes<
::8������������������������������������2
prelu_4/Neg_1�
prelu_4/Relu_1Reluprelu_4/Neg_1:y:0*
T0*N
_output_shapes<
::8������������������������������������2
prelu_4/Relu_1�
prelu_4/mulMulprelu_4/Neg:y:0prelu_4/Relu_1:activations:0*
T0*N
_output_shapes<
::8������������������������������������2
prelu_4/mul�
prelu_4/addAddV2prelu_4/Relu:activations:0prelu_4/mul:z:0*
T0*N
_output_shapes<
::8������������������������������������2
prelu_4/add�
conv3d_5/Conv3D/ReadVariableOpReadVariableOp'conv3d_5_conv3d_readvariableop_resource**
_output_shapes
:*
dtype02 
conv3d_5/Conv3D/ReadVariableOp�
conv3d_5/Conv3DConv3Dprelu_4/add:z:0&conv3d_5/Conv3D/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8������������������������������������*
paddingSAME*
strides	
2
conv3d_5/Conv3D�
conv3d_5/BiasAdd/ReadVariableOpReadVariableOp(conv3d_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv3d_5/BiasAdd/ReadVariableOp�
conv3d_5/BiasAddBiasAddconv3d_5/Conv3D:output:0'conv3d_5/BiasAdd/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8������������������������������������2
conv3d_5/BiasAdd�
prelu_5/ReluReluconv3d_5/BiasAdd:output:0*
T0*N
_output_shapes<
::8������������������������������������2
prelu_5/Relu�
prelu_5/ReadVariableOpReadVariableOpprelu_5_readvariableop_resource*&
_output_shapes
:*
dtype02
prelu_5/ReadVariableOpr
prelu_5/NegNegprelu_5/ReadVariableOp:value:0*
T0*&
_output_shapes
:2
prelu_5/Neg�
prelu_5/Neg_1Negconv3d_5/BiasAdd:output:0*
T0*N
_output_shapes<
::8������������������������������������2
prelu_5/Neg_1�
prelu_5/Relu_1Reluprelu_5/Neg_1:y:0*
T0*N
_output_shapes<
::8������������������������������������2
prelu_5/Relu_1�
prelu_5/mulMulprelu_5/Neg:y:0prelu_5/Relu_1:activations:0*
T0*N
_output_shapes<
::8������������������������������������2
prelu_5/mul�
prelu_5/addAddV2prelu_5/Relu:activations:0prelu_5/mul:z:0*
T0*N
_output_shapes<
::8������������������������������������2
prelu_5/add�
conv3d_6/Conv3D/ReadVariableOpReadVariableOp'conv3d_6_conv3d_readvariableop_resource**
_output_shapes
:*
dtype02 
conv3d_6/Conv3D/ReadVariableOp�
conv3d_6/Conv3DConv3Dprelu_5/add:z:0&conv3d_6/Conv3D/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8������������������������������������*
paddingSAME*
strides	
2
conv3d_6/Conv3D�
conv3d_6/BiasAdd/ReadVariableOpReadVariableOp(conv3d_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv3d_6/BiasAdd/ReadVariableOp�
conv3d_6/BiasAddBiasAddconv3d_6/Conv3D:output:0'conv3d_6/BiasAdd/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8������������������������������������2
conv3d_6/BiasAdd�
prelu_6/ReluReluconv3d_6/BiasAdd:output:0*
T0*N
_output_shapes<
::8������������������������������������2
prelu_6/Relu�
prelu_6/ReadVariableOpReadVariableOpprelu_6_readvariableop_resource*&
_output_shapes
:*
dtype02
prelu_6/ReadVariableOpr
prelu_6/NegNegprelu_6/ReadVariableOp:value:0*
T0*&
_output_shapes
:2
prelu_6/Neg�
prelu_6/Neg_1Negconv3d_6/BiasAdd:output:0*
T0*N
_output_shapes<
::8������������������������������������2
prelu_6/Neg_1�
prelu_6/Relu_1Reluprelu_6/Neg_1:y:0*
T0*N
_output_shapes<
::8������������������������������������2
prelu_6/Relu_1�
prelu_6/mulMulprelu_6/Neg:y:0prelu_6/Relu_1:activations:0*
T0*N
_output_shapes<
::8������������������������������������2
prelu_6/mul�
prelu_6/addAddV2prelu_6/Relu:activations:0prelu_6/mul:z:0*
T0*N
_output_shapes<
::8������������������������������������2
prelu_6/add�
conv_111/Conv3D/ReadVariableOpReadVariableOp'conv_111_conv3d_readvariableop_resource**
_output_shapes
:*
dtype02 
conv_111/Conv3D/ReadVariableOp�
conv_111/Conv3DConv3Dprelu_6/add:z:0&conv_111/Conv3D/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8������������������������������������*
paddingSAME*
strides	
2
conv_111/Conv3D�
conv_111/BiasAdd/ReadVariableOpReadVariableOp(conv_111_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv_111/BiasAdd/ReadVariableOp�
conv_111/BiasAddBiasAddconv_111/Conv3D:output:0'conv_111/BiasAdd/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8������������������������������������2
conv_111/BiasAdd�
	add_0/addAddV2conv_111/BiasAdd:output:0inputs_0*
T0*N
_output_shapes<
::8������������������������������������2
	add_0/add�
IdentityIdentityadd_0/add:z:0*
T0*N
_output_shapes<
::8������������������������������������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:8������������������������������������:8������������������������������������::::::::::::::::::::::::::::::x t
N
_output_shapes<
::8������������������������������������
"
_user_specified_name
inputs/0:xt
N
_output_shapes<
::8������������������������������������
"
_user_specified_name
inputs/1
�
k
?__inference_add_0_layer_call_and_return_conditional_losses_5693
inputs_0
inputs_1
identity�
addAddV2inputs_0inputs_1*
T0*N
_output_shapes<
::8������������������������������������2
add�
IdentityIdentityadd:z:0*
T0*N
_output_shapes<
::8������������������������������������2

Identity"
identityIdentity:output:0*�
_input_shapesv
t:8������������������������������������:8������������������������������������:x t
N
_output_shapes<
::8������������������������������������
"
_user_specified_name
inputs/0:xt
N
_output_shapes<
::8������������������������������������
"
_user_specified_name
inputs/1
�	
�
B__inference_conv3d_0_layer_call_and_return_conditional_losses_4394

inputs"
conv3d_readvariableop_resource#
biasadd_readvariableop_resource
identity��
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype02
Conv3D/ReadVariableOp�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8������������������������������������*
paddingSAME*
strides	
2
Conv3D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8������������������������������������2	
BiasAdd�
IdentityIdentityBiasAdd:output:0*
T0*N
_output_shapes<
::8������������������������������������2

Identity"
identityIdentity:output:0*U
_input_shapesD
B:8������������������������������������:::v r
N
_output_shapes<
::8������������������������������������
 
_user_specified_nameinputs
�	
�
B__inference_conv3d_6_layer_call_and_return_conditional_losses_5659

inputs"
conv3d_readvariableop_resource#
biasadd_readvariableop_resource
identity��
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype02
Conv3D/ReadVariableOp�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8������������������������������������*
paddingSAME*
strides	
2
Conv3D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8������������������������������������2	
BiasAdd�
IdentityIdentityBiasAdd:output:0*
T0*N
_output_shapes<
::8������������������������������������2

Identity"
identityIdentity:output:0*U
_input_shapesD
B:8������������������������������������:::v r
N
_output_shapes<
::8������������������������������������
 
_user_specified_nameinputs
�
l
&__inference_prelu_1_layer_call_fn_4200

inputs
unknown
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8������������������������������������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_prelu_1_layer_call_and_return_conditional_losses_41922
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*N
_output_shapes<
::8������������������������������������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:8������������������������������������:22
StatefulPartitionedCallStatefulPartitionedCall:v r
N
_output_shapes<
::8������������������������������������
 
_user_specified_nameinputs
�x
�
 __inference__traced_restore_5907
file_prefix*
&assignvariableop_conv3d_ind_0_0_kernel*
&assignvariableop_1_conv3d_ind_0_0_bias,
(assignvariableop_2_conv3d_ind_0_1_kernel*
&assignvariableop_3_conv3d_ind_0_1_bias*
&assignvariableop_4_prelu_ind_0_0_alpha*
&assignvariableop_5_prelu_ind_0_1_alpha&
"assignvariableop_6_conv3d_0_kernel$
 assignvariableop_7_conv3d_0_bias$
 assignvariableop_8_prelu_0_alpha&
"assignvariableop_9_conv3d_1_kernel%
!assignvariableop_10_conv3d_1_bias%
!assignvariableop_11_prelu_1_alpha'
#assignvariableop_12_conv3d_2_kernel%
!assignvariableop_13_conv3d_2_bias%
!assignvariableop_14_prelu_2_alpha'
#assignvariableop_15_conv3d_3_kernel%
!assignvariableop_16_conv3d_3_bias%
!assignvariableop_17_prelu_3_alpha'
#assignvariableop_18_conv3d_4_kernel%
!assignvariableop_19_conv3d_4_bias%
!assignvariableop_20_prelu_4_alpha'
#assignvariableop_21_conv3d_5_kernel%
!assignvariableop_22_conv3d_5_bias%
!assignvariableop_23_prelu_5_alpha'
#assignvariableop_24_conv3d_6_kernel%
!assignvariableop_25_conv3d_6_bias%
!assignvariableop_26_prelu_6_alpha'
#assignvariableop_27_conv_111_kernel%
!assignvariableop_28_conv_111_bias
identity_30��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/alpha/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/alpha/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/alpha/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/alpha/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/alpha/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/alpha/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-13/alpha/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-15/alpha/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-17/alpha/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*O
valueFBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapesz
x::::::::::::::::::::::::::::::*,
dtypes"
 22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOp&assignvariableop_conv3d_ind_0_0_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOp&assignvariableop_1_conv3d_ind_0_0_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOp(assignvariableop_2_conv3d_ind_0_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOp&assignvariableop_3_conv3d_ind_0_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOp&assignvariableop_4_prelu_ind_0_0_alphaIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOp&assignvariableop_5_prelu_ind_0_1_alphaIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOp"assignvariableop_6_conv3d_0_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOp assignvariableop_7_conv3d_0_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOp assignvariableop_8_prelu_0_alphaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOp"assignvariableop_9_conv3d_1_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOp!assignvariableop_10_conv3d_1_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOp!assignvariableop_11_prelu_1_alphaIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOp#assignvariableop_12_conv3d_2_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOp!assignvariableop_13_conv3d_2_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOp!assignvariableop_14_prelu_2_alphaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15�
AssignVariableOp_15AssignVariableOp#assignvariableop_15_conv3d_3_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOp!assignvariableop_16_conv3d_3_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17�
AssignVariableOp_17AssignVariableOp!assignvariableop_17_prelu_3_alphaIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18�
AssignVariableOp_18AssignVariableOp#assignvariableop_18_conv3d_4_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19�
AssignVariableOp_19AssignVariableOp!assignvariableop_19_conv3d_4_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20�
AssignVariableOp_20AssignVariableOp!assignvariableop_20_prelu_4_alphaIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21�
AssignVariableOp_21AssignVariableOp#assignvariableop_21_conv3d_5_kernelIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22�
AssignVariableOp_22AssignVariableOp!assignvariableop_22_conv3d_5_biasIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23�
AssignVariableOp_23AssignVariableOp!assignvariableop_23_prelu_5_alphaIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24�
AssignVariableOp_24AssignVariableOp#assignvariableop_24_conv3d_6_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25�
AssignVariableOp_25AssignVariableOp!assignvariableop_25_conv3d_6_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26�
AssignVariableOp_26AssignVariableOp!assignvariableop_26_prelu_6_alphaIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27�
AssignVariableOp_27AssignVariableOp#assignvariableop_27_conv_111_kernelIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28�
AssignVariableOp_28AssignVariableOp!assignvariableop_28_conv_111_biasIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_289
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�
Identity_29Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_29�
Identity_30IdentityIdentity_29:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_30"#
identity_30Identity_30:output:0*�
_input_shapesx
v: :::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�	
�
B__inference_conv3d_6_layer_call_and_return_conditional_losses_4568

inputs"
conv3d_readvariableop_resource#
biasadd_readvariableop_resource
identity��
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype02
Conv3D/ReadVariableOp�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8������������������������������������*
paddingSAME*
strides	
2
Conv3D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8������������������������������������2	
BiasAdd�
IdentityIdentityBiasAdd:output:0*
T0*N
_output_shapes<
::8������������������������������������2

Identity"
identityIdentity:output:0*U
_input_shapesD
B:8������������������������������������:::v r
N
_output_shapes<
::8������������������������������������
 
_user_specified_nameinputs
�

}
A__inference_prelu_2_layer_call_and_return_conditional_losses_4213

inputs
readvariableop_resource
identity�u
ReluReluinputs*
T0*N
_output_shapes<
::8������������������������������������2
Relu�
ReadVariableOpReadVariableOpreadvariableop_resource*&
_output_shapes
:*
dtype02
ReadVariableOpZ
NegNegReadVariableOp:value:0*
T0*&
_output_shapes
:2
Negv
Neg_1Neginputs*
T0*N
_output_shapes<
::8������������������������������������2
Neg_1|
Relu_1Relu	Neg_1:y:0*
T0*N
_output_shapes<
::8������������������������������������2
Relu_1�
mulMulNeg:y:0Relu_1:activations:0*
T0*N
_output_shapes<
::8������������������������������������2
mul�
addAddV2Relu:activations:0mul:z:0*
T0*N
_output_shapes<
::8������������������������������������2
add�
IdentityIdentityadd:z:0*
T0*N
_output_shapes<
::8������������������������������������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:8������������������������������������::v r
N
_output_shapes<
::8������������������������������������
 
_user_specified_nameinputs
��
�

G__inference_functional_15_layer_call_and_return_conditional_losses_5356
inputs_0
inputs_11
-conv3d_ind_0_1_conv3d_readvariableop_resource2
.conv3d_ind_0_1_biasadd_readvariableop_resource1
-conv3d_ind_0_0_conv3d_readvariableop_resource2
.conv3d_ind_0_0_biasadd_readvariableop_resource)
%prelu_ind_0_0_readvariableop_resource)
%prelu_ind_0_1_readvariableop_resource+
'conv3d_0_conv3d_readvariableop_resource,
(conv3d_0_biasadd_readvariableop_resource#
prelu_0_readvariableop_resource+
'conv3d_1_conv3d_readvariableop_resource,
(conv3d_1_biasadd_readvariableop_resource#
prelu_1_readvariableop_resource+
'conv3d_2_conv3d_readvariableop_resource,
(conv3d_2_biasadd_readvariableop_resource#
prelu_2_readvariableop_resource+
'conv3d_3_conv3d_readvariableop_resource,
(conv3d_3_biasadd_readvariableop_resource#
prelu_3_readvariableop_resource+
'conv3d_4_conv3d_readvariableop_resource,
(conv3d_4_biasadd_readvariableop_resource#
prelu_4_readvariableop_resource+
'conv3d_5_conv3d_readvariableop_resource,
(conv3d_5_biasadd_readvariableop_resource#
prelu_5_readvariableop_resource+
'conv3d_6_conv3d_readvariableop_resource,
(conv3d_6_biasadd_readvariableop_resource#
prelu_6_readvariableop_resource+
'conv_111_conv3d_readvariableop_resource,
(conv_111_biasadd_readvariableop_resource
identity��
$conv3d_ind_0_1/Conv3D/ReadVariableOpReadVariableOp-conv3d_ind_0_1_conv3d_readvariableop_resource**
_output_shapes
:*
dtype02&
$conv3d_ind_0_1/Conv3D/ReadVariableOp�
conv3d_ind_0_1/Conv3DConv3Dinputs_1,conv3d_ind_0_1/Conv3D/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8������������������������������������*
paddingSAME*
strides	
2
conv3d_ind_0_1/Conv3D�
%conv3d_ind_0_1/BiasAdd/ReadVariableOpReadVariableOp.conv3d_ind_0_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%conv3d_ind_0_1/BiasAdd/ReadVariableOp�
conv3d_ind_0_1/BiasAddBiasAddconv3d_ind_0_1/Conv3D:output:0-conv3d_ind_0_1/BiasAdd/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8������������������������������������2
conv3d_ind_0_1/BiasAdd�
$conv3d_ind_0_0/Conv3D/ReadVariableOpReadVariableOp-conv3d_ind_0_0_conv3d_readvariableop_resource**
_output_shapes
:*
dtype02&
$conv3d_ind_0_0/Conv3D/ReadVariableOp�
conv3d_ind_0_0/Conv3DConv3Dinputs_0,conv3d_ind_0_0/Conv3D/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8������������������������������������*
paddingSAME*
strides	
2
conv3d_ind_0_0/Conv3D�
%conv3d_ind_0_0/BiasAdd/ReadVariableOpReadVariableOp.conv3d_ind_0_0_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%conv3d_ind_0_0/BiasAdd/ReadVariableOp�
conv3d_ind_0_0/BiasAddBiasAddconv3d_ind_0_0/Conv3D:output:0-conv3d_ind_0_0/BiasAdd/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8������������������������������������2
conv3d_ind_0_0/BiasAdd�
prelu_ind_0_0/ReluReluconv3d_ind_0_0/BiasAdd:output:0*
T0*N
_output_shapes<
::8������������������������������������2
prelu_ind_0_0/Relu�
prelu_ind_0_0/ReadVariableOpReadVariableOp%prelu_ind_0_0_readvariableop_resource*&
_output_shapes
:*
dtype02
prelu_ind_0_0/ReadVariableOp�
prelu_ind_0_0/NegNeg$prelu_ind_0_0/ReadVariableOp:value:0*
T0*&
_output_shapes
:2
prelu_ind_0_0/Neg�
prelu_ind_0_0/Neg_1Negconv3d_ind_0_0/BiasAdd:output:0*
T0*N
_output_shapes<
::8������������������������������������2
prelu_ind_0_0/Neg_1�
prelu_ind_0_0/Relu_1Reluprelu_ind_0_0/Neg_1:y:0*
T0*N
_output_shapes<
::8������������������������������������2
prelu_ind_0_0/Relu_1�
prelu_ind_0_0/mulMulprelu_ind_0_0/Neg:y:0"prelu_ind_0_0/Relu_1:activations:0*
T0*N
_output_shapes<
::8������������������������������������2
prelu_ind_0_0/mul�
prelu_ind_0_0/addAddV2 prelu_ind_0_0/Relu:activations:0prelu_ind_0_0/mul:z:0*
T0*N
_output_shapes<
::8������������������������������������2
prelu_ind_0_0/add�
prelu_ind_0_1/ReluReluconv3d_ind_0_1/BiasAdd:output:0*
T0*N
_output_shapes<
::8������������������������������������2
prelu_ind_0_1/Relu�
prelu_ind_0_1/ReadVariableOpReadVariableOp%prelu_ind_0_1_readvariableop_resource*&
_output_shapes
:*
dtype02
prelu_ind_0_1/ReadVariableOp�
prelu_ind_0_1/NegNeg$prelu_ind_0_1/ReadVariableOp:value:0*
T0*&
_output_shapes
:2
prelu_ind_0_1/Neg�
prelu_ind_0_1/Neg_1Negconv3d_ind_0_1/BiasAdd:output:0*
T0*N
_output_shapes<
::8������������������������������������2
prelu_ind_0_1/Neg_1�
prelu_ind_0_1/Relu_1Reluprelu_ind_0_1/Neg_1:y:0*
T0*N
_output_shapes<
::8������������������������������������2
prelu_ind_0_1/Relu_1�
prelu_ind_0_1/mulMulprelu_ind_0_1/Neg:y:0"prelu_ind_0_1/Relu_1:activations:0*
T0*N
_output_shapes<
::8������������������������������������2
prelu_ind_0_1/mul�
prelu_ind_0_1/addAddV2 prelu_ind_0_1/Relu:activations:0prelu_ind_0_1/mul:z:0*
T0*N
_output_shapes<
::8������������������������������������2
prelu_ind_0_1/addn
concat_0/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_0/concat/axis�
concat_0/concatConcatV2prelu_ind_0_0/add:z:0prelu_ind_0_1/add:z:0concat_0/concat/axis:output:0*
N*
T0*N
_output_shapes<
::8������������������������������������2
concat_0/concat�
conv3d_0/Conv3D/ReadVariableOpReadVariableOp'conv3d_0_conv3d_readvariableop_resource**
_output_shapes
:*
dtype02 
conv3d_0/Conv3D/ReadVariableOp�
conv3d_0/Conv3DConv3Dconcat_0/concat:output:0&conv3d_0/Conv3D/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8������������������������������������*
paddingSAME*
strides	
2
conv3d_0/Conv3D�
conv3d_0/BiasAdd/ReadVariableOpReadVariableOp(conv3d_0_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv3d_0/BiasAdd/ReadVariableOp�
conv3d_0/BiasAddBiasAddconv3d_0/Conv3D:output:0'conv3d_0/BiasAdd/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8������������������������������������2
conv3d_0/BiasAdd�
prelu_0/ReluReluconv3d_0/BiasAdd:output:0*
T0*N
_output_shapes<
::8������������������������������������2
prelu_0/Relu�
prelu_0/ReadVariableOpReadVariableOpprelu_0_readvariableop_resource*&
_output_shapes
:*
dtype02
prelu_0/ReadVariableOpr
prelu_0/NegNegprelu_0/ReadVariableOp:value:0*
T0*&
_output_shapes
:2
prelu_0/Neg�
prelu_0/Neg_1Negconv3d_0/BiasAdd:output:0*
T0*N
_output_shapes<
::8������������������������������������2
prelu_0/Neg_1�
prelu_0/Relu_1Reluprelu_0/Neg_1:y:0*
T0*N
_output_shapes<
::8������������������������������������2
prelu_0/Relu_1�
prelu_0/mulMulprelu_0/Neg:y:0prelu_0/Relu_1:activations:0*
T0*N
_output_shapes<
::8������������������������������������2
prelu_0/mul�
prelu_0/addAddV2prelu_0/Relu:activations:0prelu_0/mul:z:0*
T0*N
_output_shapes<
::8������������������������������������2
prelu_0/add�
conv3d_1/Conv3D/ReadVariableOpReadVariableOp'conv3d_1_conv3d_readvariableop_resource**
_output_shapes
:*
dtype02 
conv3d_1/Conv3D/ReadVariableOp�
conv3d_1/Conv3DConv3Dprelu_0/add:z:0&conv3d_1/Conv3D/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8������������������������������������*
paddingSAME*
strides	
2
conv3d_1/Conv3D�
conv3d_1/BiasAdd/ReadVariableOpReadVariableOp(conv3d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv3d_1/BiasAdd/ReadVariableOp�
conv3d_1/BiasAddBiasAddconv3d_1/Conv3D:output:0'conv3d_1/BiasAdd/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8������������������������������������2
conv3d_1/BiasAdd�
prelu_1/ReluReluconv3d_1/BiasAdd:output:0*
T0*N
_output_shapes<
::8������������������������������������2
prelu_1/Relu�
prelu_1/ReadVariableOpReadVariableOpprelu_1_readvariableop_resource*&
_output_shapes
:*
dtype02
prelu_1/ReadVariableOpr
prelu_1/NegNegprelu_1/ReadVariableOp:value:0*
T0*&
_output_shapes
:2
prelu_1/Neg�
prelu_1/Neg_1Negconv3d_1/BiasAdd:output:0*
T0*N
_output_shapes<
::8������������������������������������2
prelu_1/Neg_1�
prelu_1/Relu_1Reluprelu_1/Neg_1:y:0*
T0*N
_output_shapes<
::8������������������������������������2
prelu_1/Relu_1�
prelu_1/mulMulprelu_1/Neg:y:0prelu_1/Relu_1:activations:0*
T0*N
_output_shapes<
::8������������������������������������2
prelu_1/mul�
prelu_1/addAddV2prelu_1/Relu:activations:0prelu_1/mul:z:0*
T0*N
_output_shapes<
::8������������������������������������2
prelu_1/add�
conv3d_2/Conv3D/ReadVariableOpReadVariableOp'conv3d_2_conv3d_readvariableop_resource**
_output_shapes
:*
dtype02 
conv3d_2/Conv3D/ReadVariableOp�
conv3d_2/Conv3DConv3Dprelu_1/add:z:0&conv3d_2/Conv3D/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8������������������������������������*
paddingSAME*
strides	
2
conv3d_2/Conv3D�
conv3d_2/BiasAdd/ReadVariableOpReadVariableOp(conv3d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv3d_2/BiasAdd/ReadVariableOp�
conv3d_2/BiasAddBiasAddconv3d_2/Conv3D:output:0'conv3d_2/BiasAdd/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8������������������������������������2
conv3d_2/BiasAdd�
prelu_2/ReluReluconv3d_2/BiasAdd:output:0*
T0*N
_output_shapes<
::8������������������������������������2
prelu_2/Relu�
prelu_2/ReadVariableOpReadVariableOpprelu_2_readvariableop_resource*&
_output_shapes
:*
dtype02
prelu_2/ReadVariableOpr
prelu_2/NegNegprelu_2/ReadVariableOp:value:0*
T0*&
_output_shapes
:2
prelu_2/Neg�
prelu_2/Neg_1Negconv3d_2/BiasAdd:output:0*
T0*N
_output_shapes<
::8������������������������������������2
prelu_2/Neg_1�
prelu_2/Relu_1Reluprelu_2/Neg_1:y:0*
T0*N
_output_shapes<
::8������������������������������������2
prelu_2/Relu_1�
prelu_2/mulMulprelu_2/Neg:y:0prelu_2/Relu_1:activations:0*
T0*N
_output_shapes<
::8������������������������������������2
prelu_2/mul�
prelu_2/addAddV2prelu_2/Relu:activations:0prelu_2/mul:z:0*
T0*N
_output_shapes<
::8������������������������������������2
prelu_2/add�
conv3d_3/Conv3D/ReadVariableOpReadVariableOp'conv3d_3_conv3d_readvariableop_resource**
_output_shapes
:*
dtype02 
conv3d_3/Conv3D/ReadVariableOp�
conv3d_3/Conv3DConv3Dprelu_2/add:z:0&conv3d_3/Conv3D/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8������������������������������������*
paddingSAME*
strides	
2
conv3d_3/Conv3D�
conv3d_3/BiasAdd/ReadVariableOpReadVariableOp(conv3d_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv3d_3/BiasAdd/ReadVariableOp�
conv3d_3/BiasAddBiasAddconv3d_3/Conv3D:output:0'conv3d_3/BiasAdd/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8������������������������������������2
conv3d_3/BiasAdd�
prelu_3/ReluReluconv3d_3/BiasAdd:output:0*
T0*N
_output_shapes<
::8������������������������������������2
prelu_3/Relu�
prelu_3/ReadVariableOpReadVariableOpprelu_3_readvariableop_resource*&
_output_shapes
:*
dtype02
prelu_3/ReadVariableOpr
prelu_3/NegNegprelu_3/ReadVariableOp:value:0*
T0*&
_output_shapes
:2
prelu_3/Neg�
prelu_3/Neg_1Negconv3d_3/BiasAdd:output:0*
T0*N
_output_shapes<
::8������������������������������������2
prelu_3/Neg_1�
prelu_3/Relu_1Reluprelu_3/Neg_1:y:0*
T0*N
_output_shapes<
::8������������������������������������2
prelu_3/Relu_1�
prelu_3/mulMulprelu_3/Neg:y:0prelu_3/Relu_1:activations:0*
T0*N
_output_shapes<
::8������������������������������������2
prelu_3/mul�
prelu_3/addAddV2prelu_3/Relu:activations:0prelu_3/mul:z:0*
T0*N
_output_shapes<
::8������������������������������������2
prelu_3/add�
conv3d_4/Conv3D/ReadVariableOpReadVariableOp'conv3d_4_conv3d_readvariableop_resource**
_output_shapes
:*
dtype02 
conv3d_4/Conv3D/ReadVariableOp�
conv3d_4/Conv3DConv3Dprelu_3/add:z:0&conv3d_4/Conv3D/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8������������������������������������*
paddingSAME*
strides	
2
conv3d_4/Conv3D�
conv3d_4/BiasAdd/ReadVariableOpReadVariableOp(conv3d_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv3d_4/BiasAdd/ReadVariableOp�
conv3d_4/BiasAddBiasAddconv3d_4/Conv3D:output:0'conv3d_4/BiasAdd/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8������������������������������������2
conv3d_4/BiasAdd�
prelu_4/ReluReluconv3d_4/BiasAdd:output:0*
T0*N
_output_shapes<
::8������������������������������������2
prelu_4/Relu�
prelu_4/ReadVariableOpReadVariableOpprelu_4_readvariableop_resource*&
_output_shapes
:*
dtype02
prelu_4/ReadVariableOpr
prelu_4/NegNegprelu_4/ReadVariableOp:value:0*
T0*&
_output_shapes
:2
prelu_4/Neg�
prelu_4/Neg_1Negconv3d_4/BiasAdd:output:0*
T0*N
_output_shapes<
::8������������������������������������2
prelu_4/Neg_1�
prelu_4/Relu_1Reluprelu_4/Neg_1:y:0*
T0*N
_output_shapes<
::8������������������������������������2
prelu_4/Relu_1�
prelu_4/mulMulprelu_4/Neg:y:0prelu_4/Relu_1:activations:0*
T0*N
_output_shapes<
::8������������������������������������2
prelu_4/mul�
prelu_4/addAddV2prelu_4/Relu:activations:0prelu_4/mul:z:0*
T0*N
_output_shapes<
::8������������������������������������2
prelu_4/add�
conv3d_5/Conv3D/ReadVariableOpReadVariableOp'conv3d_5_conv3d_readvariableop_resource**
_output_shapes
:*
dtype02 
conv3d_5/Conv3D/ReadVariableOp�
conv3d_5/Conv3DConv3Dprelu_4/add:z:0&conv3d_5/Conv3D/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8������������������������������������*
paddingSAME*
strides	
2
conv3d_5/Conv3D�
conv3d_5/BiasAdd/ReadVariableOpReadVariableOp(conv3d_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv3d_5/BiasAdd/ReadVariableOp�
conv3d_5/BiasAddBiasAddconv3d_5/Conv3D:output:0'conv3d_5/BiasAdd/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8������������������������������������2
conv3d_5/BiasAdd�
prelu_5/ReluReluconv3d_5/BiasAdd:output:0*
T0*N
_output_shapes<
::8������������������������������������2
prelu_5/Relu�
prelu_5/ReadVariableOpReadVariableOpprelu_5_readvariableop_resource*&
_output_shapes
:*
dtype02
prelu_5/ReadVariableOpr
prelu_5/NegNegprelu_5/ReadVariableOp:value:0*
T0*&
_output_shapes
:2
prelu_5/Neg�
prelu_5/Neg_1Negconv3d_5/BiasAdd:output:0*
T0*N
_output_shapes<
::8������������������������������������2
prelu_5/Neg_1�
prelu_5/Relu_1Reluprelu_5/Neg_1:y:0*
T0*N
_output_shapes<
::8������������������������������������2
prelu_5/Relu_1�
prelu_5/mulMulprelu_5/Neg:y:0prelu_5/Relu_1:activations:0*
T0*N
_output_shapes<
::8������������������������������������2
prelu_5/mul�
prelu_5/addAddV2prelu_5/Relu:activations:0prelu_5/mul:z:0*
T0*N
_output_shapes<
::8������������������������������������2
prelu_5/add�
conv3d_6/Conv3D/ReadVariableOpReadVariableOp'conv3d_6_conv3d_readvariableop_resource**
_output_shapes
:*
dtype02 
conv3d_6/Conv3D/ReadVariableOp�
conv3d_6/Conv3DConv3Dprelu_5/add:z:0&conv3d_6/Conv3D/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8������������������������������������*
paddingSAME*
strides	
2
conv3d_6/Conv3D�
conv3d_6/BiasAdd/ReadVariableOpReadVariableOp(conv3d_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv3d_6/BiasAdd/ReadVariableOp�
conv3d_6/BiasAddBiasAddconv3d_6/Conv3D:output:0'conv3d_6/BiasAdd/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8������������������������������������2
conv3d_6/BiasAdd�
prelu_6/ReluReluconv3d_6/BiasAdd:output:0*
T0*N
_output_shapes<
::8������������������������������������2
prelu_6/Relu�
prelu_6/ReadVariableOpReadVariableOpprelu_6_readvariableop_resource*&
_output_shapes
:*
dtype02
prelu_6/ReadVariableOpr
prelu_6/NegNegprelu_6/ReadVariableOp:value:0*
T0*&
_output_shapes
:2
prelu_6/Neg�
prelu_6/Neg_1Negconv3d_6/BiasAdd:output:0*
T0*N
_output_shapes<
::8������������������������������������2
prelu_6/Neg_1�
prelu_6/Relu_1Reluprelu_6/Neg_1:y:0*
T0*N
_output_shapes<
::8������������������������������������2
prelu_6/Relu_1�
prelu_6/mulMulprelu_6/Neg:y:0prelu_6/Relu_1:activations:0*
T0*N
_output_shapes<
::8������������������������������������2
prelu_6/mul�
prelu_6/addAddV2prelu_6/Relu:activations:0prelu_6/mul:z:0*
T0*N
_output_shapes<
::8������������������������������������2
prelu_6/add�
conv_111/Conv3D/ReadVariableOpReadVariableOp'conv_111_conv3d_readvariableop_resource**
_output_shapes
:*
dtype02 
conv_111/Conv3D/ReadVariableOp�
conv_111/Conv3DConv3Dprelu_6/add:z:0&conv_111/Conv3D/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8������������������������������������*
paddingSAME*
strides	
2
conv_111/Conv3D�
conv_111/BiasAdd/ReadVariableOpReadVariableOp(conv_111_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv_111/BiasAdd/ReadVariableOp�
conv_111/BiasAddBiasAddconv_111/Conv3D:output:0'conv_111/BiasAdd/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8������������������������������������2
conv_111/BiasAdd�
	add_0/addAddV2conv_111/BiasAdd:output:0inputs_0*
T0*N
_output_shapes<
::8������������������������������������2
	add_0/add�
IdentityIdentityadd_0/add:z:0*
T0*N
_output_shapes<
::8������������������������������������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:8������������������������������������:8������������������������������������::::::::::::::::::::::::::::::x t
N
_output_shapes<
::8������������������������������������
"
_user_specified_name
inputs/0:xt
N
_output_shapes<
::8������������������������������������
"
_user_specified_name
inputs/1
�
S
'__inference_concat_0_layer_call_fn_5535
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8������������������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_concat_0_layer_call_and_return_conditional_losses_43752
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*N
_output_shapes<
::8������������������������������������2

Identity"
identityIdentity:output:0*�
_input_shapesv
t:8������������������������������������:8������������������������������������:x t
N
_output_shapes<
::8������������������������������������
"
_user_specified_name
inputs/0:xt
N
_output_shapes<
::8������������������������������������
"
_user_specified_name
inputs/1
��
�
__inference__wrapped_model_4116
input_0
input_1?
;functional_15_conv3d_ind_0_1_conv3d_readvariableop_resource@
<functional_15_conv3d_ind_0_1_biasadd_readvariableop_resource?
;functional_15_conv3d_ind_0_0_conv3d_readvariableop_resource@
<functional_15_conv3d_ind_0_0_biasadd_readvariableop_resource7
3functional_15_prelu_ind_0_0_readvariableop_resource7
3functional_15_prelu_ind_0_1_readvariableop_resource9
5functional_15_conv3d_0_conv3d_readvariableop_resource:
6functional_15_conv3d_0_biasadd_readvariableop_resource1
-functional_15_prelu_0_readvariableop_resource9
5functional_15_conv3d_1_conv3d_readvariableop_resource:
6functional_15_conv3d_1_biasadd_readvariableop_resource1
-functional_15_prelu_1_readvariableop_resource9
5functional_15_conv3d_2_conv3d_readvariableop_resource:
6functional_15_conv3d_2_biasadd_readvariableop_resource1
-functional_15_prelu_2_readvariableop_resource9
5functional_15_conv3d_3_conv3d_readvariableop_resource:
6functional_15_conv3d_3_biasadd_readvariableop_resource1
-functional_15_prelu_3_readvariableop_resource9
5functional_15_conv3d_4_conv3d_readvariableop_resource:
6functional_15_conv3d_4_biasadd_readvariableop_resource1
-functional_15_prelu_4_readvariableop_resource9
5functional_15_conv3d_5_conv3d_readvariableop_resource:
6functional_15_conv3d_5_biasadd_readvariableop_resource1
-functional_15_prelu_5_readvariableop_resource9
5functional_15_conv3d_6_conv3d_readvariableop_resource:
6functional_15_conv3d_6_biasadd_readvariableop_resource1
-functional_15_prelu_6_readvariableop_resource9
5functional_15_conv_111_conv3d_readvariableop_resource:
6functional_15_conv_111_biasadd_readvariableop_resource
identity��
2functional_15/conv3d_ind_0_1/Conv3D/ReadVariableOpReadVariableOp;functional_15_conv3d_ind_0_1_conv3d_readvariableop_resource**
_output_shapes
:*
dtype024
2functional_15/conv3d_ind_0_1/Conv3D/ReadVariableOp�
#functional_15/conv3d_ind_0_1/Conv3DConv3Dinput_1:functional_15/conv3d_ind_0_1/Conv3D/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8������������������������������������*
paddingSAME*
strides	
2%
#functional_15/conv3d_ind_0_1/Conv3D�
3functional_15/conv3d_ind_0_1/BiasAdd/ReadVariableOpReadVariableOp<functional_15_conv3d_ind_0_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype025
3functional_15/conv3d_ind_0_1/BiasAdd/ReadVariableOp�
$functional_15/conv3d_ind_0_1/BiasAddBiasAdd,functional_15/conv3d_ind_0_1/Conv3D:output:0;functional_15/conv3d_ind_0_1/BiasAdd/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8������������������������������������2&
$functional_15/conv3d_ind_0_1/BiasAdd�
2functional_15/conv3d_ind_0_0/Conv3D/ReadVariableOpReadVariableOp;functional_15_conv3d_ind_0_0_conv3d_readvariableop_resource**
_output_shapes
:*
dtype024
2functional_15/conv3d_ind_0_0/Conv3D/ReadVariableOp�
#functional_15/conv3d_ind_0_0/Conv3DConv3Dinput_0:functional_15/conv3d_ind_0_0/Conv3D/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8������������������������������������*
paddingSAME*
strides	
2%
#functional_15/conv3d_ind_0_0/Conv3D�
3functional_15/conv3d_ind_0_0/BiasAdd/ReadVariableOpReadVariableOp<functional_15_conv3d_ind_0_0_biasadd_readvariableop_resource*
_output_shapes
:*
dtype025
3functional_15/conv3d_ind_0_0/BiasAdd/ReadVariableOp�
$functional_15/conv3d_ind_0_0/BiasAddBiasAdd,functional_15/conv3d_ind_0_0/Conv3D:output:0;functional_15/conv3d_ind_0_0/BiasAdd/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8������������������������������������2&
$functional_15/conv3d_ind_0_0/BiasAdd�
 functional_15/prelu_ind_0_0/ReluRelu-functional_15/conv3d_ind_0_0/BiasAdd:output:0*
T0*N
_output_shapes<
::8������������������������������������2"
 functional_15/prelu_ind_0_0/Relu�
*functional_15/prelu_ind_0_0/ReadVariableOpReadVariableOp3functional_15_prelu_ind_0_0_readvariableop_resource*&
_output_shapes
:*
dtype02,
*functional_15/prelu_ind_0_0/ReadVariableOp�
functional_15/prelu_ind_0_0/NegNeg2functional_15/prelu_ind_0_0/ReadVariableOp:value:0*
T0*&
_output_shapes
:2!
functional_15/prelu_ind_0_0/Neg�
!functional_15/prelu_ind_0_0/Neg_1Neg-functional_15/conv3d_ind_0_0/BiasAdd:output:0*
T0*N
_output_shapes<
::8������������������������������������2#
!functional_15/prelu_ind_0_0/Neg_1�
"functional_15/prelu_ind_0_0/Relu_1Relu%functional_15/prelu_ind_0_0/Neg_1:y:0*
T0*N
_output_shapes<
::8������������������������������������2$
"functional_15/prelu_ind_0_0/Relu_1�
functional_15/prelu_ind_0_0/mulMul#functional_15/prelu_ind_0_0/Neg:y:00functional_15/prelu_ind_0_0/Relu_1:activations:0*
T0*N
_output_shapes<
::8������������������������������������2!
functional_15/prelu_ind_0_0/mul�
functional_15/prelu_ind_0_0/addAddV2.functional_15/prelu_ind_0_0/Relu:activations:0#functional_15/prelu_ind_0_0/mul:z:0*
T0*N
_output_shapes<
::8������������������������������������2!
functional_15/prelu_ind_0_0/add�
 functional_15/prelu_ind_0_1/ReluRelu-functional_15/conv3d_ind_0_1/BiasAdd:output:0*
T0*N
_output_shapes<
::8������������������������������������2"
 functional_15/prelu_ind_0_1/Relu�
*functional_15/prelu_ind_0_1/ReadVariableOpReadVariableOp3functional_15_prelu_ind_0_1_readvariableop_resource*&
_output_shapes
:*
dtype02,
*functional_15/prelu_ind_0_1/ReadVariableOp�
functional_15/prelu_ind_0_1/NegNeg2functional_15/prelu_ind_0_1/ReadVariableOp:value:0*
T0*&
_output_shapes
:2!
functional_15/prelu_ind_0_1/Neg�
!functional_15/prelu_ind_0_1/Neg_1Neg-functional_15/conv3d_ind_0_1/BiasAdd:output:0*
T0*N
_output_shapes<
::8������������������������������������2#
!functional_15/prelu_ind_0_1/Neg_1�
"functional_15/prelu_ind_0_1/Relu_1Relu%functional_15/prelu_ind_0_1/Neg_1:y:0*
T0*N
_output_shapes<
::8������������������������������������2$
"functional_15/prelu_ind_0_1/Relu_1�
functional_15/prelu_ind_0_1/mulMul#functional_15/prelu_ind_0_1/Neg:y:00functional_15/prelu_ind_0_1/Relu_1:activations:0*
T0*N
_output_shapes<
::8������������������������������������2!
functional_15/prelu_ind_0_1/mul�
functional_15/prelu_ind_0_1/addAddV2.functional_15/prelu_ind_0_1/Relu:activations:0#functional_15/prelu_ind_0_1/mul:z:0*
T0*N
_output_shapes<
::8������������������������������������2!
functional_15/prelu_ind_0_1/add�
"functional_15/concat_0/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2$
"functional_15/concat_0/concat/axis�
functional_15/concat_0/concatConcatV2#functional_15/prelu_ind_0_0/add:z:0#functional_15/prelu_ind_0_1/add:z:0+functional_15/concat_0/concat/axis:output:0*
N*
T0*N
_output_shapes<
::8������������������������������������2
functional_15/concat_0/concat�
,functional_15/conv3d_0/Conv3D/ReadVariableOpReadVariableOp5functional_15_conv3d_0_conv3d_readvariableop_resource**
_output_shapes
:*
dtype02.
,functional_15/conv3d_0/Conv3D/ReadVariableOp�
functional_15/conv3d_0/Conv3DConv3D&functional_15/concat_0/concat:output:04functional_15/conv3d_0/Conv3D/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8������������������������������������*
paddingSAME*
strides	
2
functional_15/conv3d_0/Conv3D�
-functional_15/conv3d_0/BiasAdd/ReadVariableOpReadVariableOp6functional_15_conv3d_0_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-functional_15/conv3d_0/BiasAdd/ReadVariableOp�
functional_15/conv3d_0/BiasAddBiasAdd&functional_15/conv3d_0/Conv3D:output:05functional_15/conv3d_0/BiasAdd/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8������������������������������������2 
functional_15/conv3d_0/BiasAdd�
functional_15/prelu_0/ReluRelu'functional_15/conv3d_0/BiasAdd:output:0*
T0*N
_output_shapes<
::8������������������������������������2
functional_15/prelu_0/Relu�
$functional_15/prelu_0/ReadVariableOpReadVariableOp-functional_15_prelu_0_readvariableop_resource*&
_output_shapes
:*
dtype02&
$functional_15/prelu_0/ReadVariableOp�
functional_15/prelu_0/NegNeg,functional_15/prelu_0/ReadVariableOp:value:0*
T0*&
_output_shapes
:2
functional_15/prelu_0/Neg�
functional_15/prelu_0/Neg_1Neg'functional_15/conv3d_0/BiasAdd:output:0*
T0*N
_output_shapes<
::8������������������������������������2
functional_15/prelu_0/Neg_1�
functional_15/prelu_0/Relu_1Relufunctional_15/prelu_0/Neg_1:y:0*
T0*N
_output_shapes<
::8������������������������������������2
functional_15/prelu_0/Relu_1�
functional_15/prelu_0/mulMulfunctional_15/prelu_0/Neg:y:0*functional_15/prelu_0/Relu_1:activations:0*
T0*N
_output_shapes<
::8������������������������������������2
functional_15/prelu_0/mul�
functional_15/prelu_0/addAddV2(functional_15/prelu_0/Relu:activations:0functional_15/prelu_0/mul:z:0*
T0*N
_output_shapes<
::8������������������������������������2
functional_15/prelu_0/add�
,functional_15/conv3d_1/Conv3D/ReadVariableOpReadVariableOp5functional_15_conv3d_1_conv3d_readvariableop_resource**
_output_shapes
:*
dtype02.
,functional_15/conv3d_1/Conv3D/ReadVariableOp�
functional_15/conv3d_1/Conv3DConv3Dfunctional_15/prelu_0/add:z:04functional_15/conv3d_1/Conv3D/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8������������������������������������*
paddingSAME*
strides	
2
functional_15/conv3d_1/Conv3D�
-functional_15/conv3d_1/BiasAdd/ReadVariableOpReadVariableOp6functional_15_conv3d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-functional_15/conv3d_1/BiasAdd/ReadVariableOp�
functional_15/conv3d_1/BiasAddBiasAdd&functional_15/conv3d_1/Conv3D:output:05functional_15/conv3d_1/BiasAdd/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8������������������������������������2 
functional_15/conv3d_1/BiasAdd�
functional_15/prelu_1/ReluRelu'functional_15/conv3d_1/BiasAdd:output:0*
T0*N
_output_shapes<
::8������������������������������������2
functional_15/prelu_1/Relu�
$functional_15/prelu_1/ReadVariableOpReadVariableOp-functional_15_prelu_1_readvariableop_resource*&
_output_shapes
:*
dtype02&
$functional_15/prelu_1/ReadVariableOp�
functional_15/prelu_1/NegNeg,functional_15/prelu_1/ReadVariableOp:value:0*
T0*&
_output_shapes
:2
functional_15/prelu_1/Neg�
functional_15/prelu_1/Neg_1Neg'functional_15/conv3d_1/BiasAdd:output:0*
T0*N
_output_shapes<
::8������������������������������������2
functional_15/prelu_1/Neg_1�
functional_15/prelu_1/Relu_1Relufunctional_15/prelu_1/Neg_1:y:0*
T0*N
_output_shapes<
::8������������������������������������2
functional_15/prelu_1/Relu_1�
functional_15/prelu_1/mulMulfunctional_15/prelu_1/Neg:y:0*functional_15/prelu_1/Relu_1:activations:0*
T0*N
_output_shapes<
::8������������������������������������2
functional_15/prelu_1/mul�
functional_15/prelu_1/addAddV2(functional_15/prelu_1/Relu:activations:0functional_15/prelu_1/mul:z:0*
T0*N
_output_shapes<
::8������������������������������������2
functional_15/prelu_1/add�
,functional_15/conv3d_2/Conv3D/ReadVariableOpReadVariableOp5functional_15_conv3d_2_conv3d_readvariableop_resource**
_output_shapes
:*
dtype02.
,functional_15/conv3d_2/Conv3D/ReadVariableOp�
functional_15/conv3d_2/Conv3DConv3Dfunctional_15/prelu_1/add:z:04functional_15/conv3d_2/Conv3D/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8������������������������������������*
paddingSAME*
strides	
2
functional_15/conv3d_2/Conv3D�
-functional_15/conv3d_2/BiasAdd/ReadVariableOpReadVariableOp6functional_15_conv3d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-functional_15/conv3d_2/BiasAdd/ReadVariableOp�
functional_15/conv3d_2/BiasAddBiasAdd&functional_15/conv3d_2/Conv3D:output:05functional_15/conv3d_2/BiasAdd/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8������������������������������������2 
functional_15/conv3d_2/BiasAdd�
functional_15/prelu_2/ReluRelu'functional_15/conv3d_2/BiasAdd:output:0*
T0*N
_output_shapes<
::8������������������������������������2
functional_15/prelu_2/Relu�
$functional_15/prelu_2/ReadVariableOpReadVariableOp-functional_15_prelu_2_readvariableop_resource*&
_output_shapes
:*
dtype02&
$functional_15/prelu_2/ReadVariableOp�
functional_15/prelu_2/NegNeg,functional_15/prelu_2/ReadVariableOp:value:0*
T0*&
_output_shapes
:2
functional_15/prelu_2/Neg�
functional_15/prelu_2/Neg_1Neg'functional_15/conv3d_2/BiasAdd:output:0*
T0*N
_output_shapes<
::8������������������������������������2
functional_15/prelu_2/Neg_1�
functional_15/prelu_2/Relu_1Relufunctional_15/prelu_2/Neg_1:y:0*
T0*N
_output_shapes<
::8������������������������������������2
functional_15/prelu_2/Relu_1�
functional_15/prelu_2/mulMulfunctional_15/prelu_2/Neg:y:0*functional_15/prelu_2/Relu_1:activations:0*
T0*N
_output_shapes<
::8������������������������������������2
functional_15/prelu_2/mul�
functional_15/prelu_2/addAddV2(functional_15/prelu_2/Relu:activations:0functional_15/prelu_2/mul:z:0*
T0*N
_output_shapes<
::8������������������������������������2
functional_15/prelu_2/add�
,functional_15/conv3d_3/Conv3D/ReadVariableOpReadVariableOp5functional_15_conv3d_3_conv3d_readvariableop_resource**
_output_shapes
:*
dtype02.
,functional_15/conv3d_3/Conv3D/ReadVariableOp�
functional_15/conv3d_3/Conv3DConv3Dfunctional_15/prelu_2/add:z:04functional_15/conv3d_3/Conv3D/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8������������������������������������*
paddingSAME*
strides	
2
functional_15/conv3d_3/Conv3D�
-functional_15/conv3d_3/BiasAdd/ReadVariableOpReadVariableOp6functional_15_conv3d_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-functional_15/conv3d_3/BiasAdd/ReadVariableOp�
functional_15/conv3d_3/BiasAddBiasAdd&functional_15/conv3d_3/Conv3D:output:05functional_15/conv3d_3/BiasAdd/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8������������������������������������2 
functional_15/conv3d_3/BiasAdd�
functional_15/prelu_3/ReluRelu'functional_15/conv3d_3/BiasAdd:output:0*
T0*N
_output_shapes<
::8������������������������������������2
functional_15/prelu_3/Relu�
$functional_15/prelu_3/ReadVariableOpReadVariableOp-functional_15_prelu_3_readvariableop_resource*&
_output_shapes
:*
dtype02&
$functional_15/prelu_3/ReadVariableOp�
functional_15/prelu_3/NegNeg,functional_15/prelu_3/ReadVariableOp:value:0*
T0*&
_output_shapes
:2
functional_15/prelu_3/Neg�
functional_15/prelu_3/Neg_1Neg'functional_15/conv3d_3/BiasAdd:output:0*
T0*N
_output_shapes<
::8������������������������������������2
functional_15/prelu_3/Neg_1�
functional_15/prelu_3/Relu_1Relufunctional_15/prelu_3/Neg_1:y:0*
T0*N
_output_shapes<
::8������������������������������������2
functional_15/prelu_3/Relu_1�
functional_15/prelu_3/mulMulfunctional_15/prelu_3/Neg:y:0*functional_15/prelu_3/Relu_1:activations:0*
T0*N
_output_shapes<
::8������������������������������������2
functional_15/prelu_3/mul�
functional_15/prelu_3/addAddV2(functional_15/prelu_3/Relu:activations:0functional_15/prelu_3/mul:z:0*
T0*N
_output_shapes<
::8������������������������������������2
functional_15/prelu_3/add�
,functional_15/conv3d_4/Conv3D/ReadVariableOpReadVariableOp5functional_15_conv3d_4_conv3d_readvariableop_resource**
_output_shapes
:*
dtype02.
,functional_15/conv3d_4/Conv3D/ReadVariableOp�
functional_15/conv3d_4/Conv3DConv3Dfunctional_15/prelu_3/add:z:04functional_15/conv3d_4/Conv3D/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8������������������������������������*
paddingSAME*
strides	
2
functional_15/conv3d_4/Conv3D�
-functional_15/conv3d_4/BiasAdd/ReadVariableOpReadVariableOp6functional_15_conv3d_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-functional_15/conv3d_4/BiasAdd/ReadVariableOp�
functional_15/conv3d_4/BiasAddBiasAdd&functional_15/conv3d_4/Conv3D:output:05functional_15/conv3d_4/BiasAdd/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8������������������������������������2 
functional_15/conv3d_4/BiasAdd�
functional_15/prelu_4/ReluRelu'functional_15/conv3d_4/BiasAdd:output:0*
T0*N
_output_shapes<
::8������������������������������������2
functional_15/prelu_4/Relu�
$functional_15/prelu_4/ReadVariableOpReadVariableOp-functional_15_prelu_4_readvariableop_resource*&
_output_shapes
:*
dtype02&
$functional_15/prelu_4/ReadVariableOp�
functional_15/prelu_4/NegNeg,functional_15/prelu_4/ReadVariableOp:value:0*
T0*&
_output_shapes
:2
functional_15/prelu_4/Neg�
functional_15/prelu_4/Neg_1Neg'functional_15/conv3d_4/BiasAdd:output:0*
T0*N
_output_shapes<
::8������������������������������������2
functional_15/prelu_4/Neg_1�
functional_15/prelu_4/Relu_1Relufunctional_15/prelu_4/Neg_1:y:0*
T0*N
_output_shapes<
::8������������������������������������2
functional_15/prelu_4/Relu_1�
functional_15/prelu_4/mulMulfunctional_15/prelu_4/Neg:y:0*functional_15/prelu_4/Relu_1:activations:0*
T0*N
_output_shapes<
::8������������������������������������2
functional_15/prelu_4/mul�
functional_15/prelu_4/addAddV2(functional_15/prelu_4/Relu:activations:0functional_15/prelu_4/mul:z:0*
T0*N
_output_shapes<
::8������������������������������������2
functional_15/prelu_4/add�
,functional_15/conv3d_5/Conv3D/ReadVariableOpReadVariableOp5functional_15_conv3d_5_conv3d_readvariableop_resource**
_output_shapes
:*
dtype02.
,functional_15/conv3d_5/Conv3D/ReadVariableOp�
functional_15/conv3d_5/Conv3DConv3Dfunctional_15/prelu_4/add:z:04functional_15/conv3d_5/Conv3D/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8������������������������������������*
paddingSAME*
strides	
2
functional_15/conv3d_5/Conv3D�
-functional_15/conv3d_5/BiasAdd/ReadVariableOpReadVariableOp6functional_15_conv3d_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-functional_15/conv3d_5/BiasAdd/ReadVariableOp�
functional_15/conv3d_5/BiasAddBiasAdd&functional_15/conv3d_5/Conv3D:output:05functional_15/conv3d_5/BiasAdd/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8������������������������������������2 
functional_15/conv3d_5/BiasAdd�
functional_15/prelu_5/ReluRelu'functional_15/conv3d_5/BiasAdd:output:0*
T0*N
_output_shapes<
::8������������������������������������2
functional_15/prelu_5/Relu�
$functional_15/prelu_5/ReadVariableOpReadVariableOp-functional_15_prelu_5_readvariableop_resource*&
_output_shapes
:*
dtype02&
$functional_15/prelu_5/ReadVariableOp�
functional_15/prelu_5/NegNeg,functional_15/prelu_5/ReadVariableOp:value:0*
T0*&
_output_shapes
:2
functional_15/prelu_5/Neg�
functional_15/prelu_5/Neg_1Neg'functional_15/conv3d_5/BiasAdd:output:0*
T0*N
_output_shapes<
::8������������������������������������2
functional_15/prelu_5/Neg_1�
functional_15/prelu_5/Relu_1Relufunctional_15/prelu_5/Neg_1:y:0*
T0*N
_output_shapes<
::8������������������������������������2
functional_15/prelu_5/Relu_1�
functional_15/prelu_5/mulMulfunctional_15/prelu_5/Neg:y:0*functional_15/prelu_5/Relu_1:activations:0*
T0*N
_output_shapes<
::8������������������������������������2
functional_15/prelu_5/mul�
functional_15/prelu_5/addAddV2(functional_15/prelu_5/Relu:activations:0functional_15/prelu_5/mul:z:0*
T0*N
_output_shapes<
::8������������������������������������2
functional_15/prelu_5/add�
,functional_15/conv3d_6/Conv3D/ReadVariableOpReadVariableOp5functional_15_conv3d_6_conv3d_readvariableop_resource**
_output_shapes
:*
dtype02.
,functional_15/conv3d_6/Conv3D/ReadVariableOp�
functional_15/conv3d_6/Conv3DConv3Dfunctional_15/prelu_5/add:z:04functional_15/conv3d_6/Conv3D/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8������������������������������������*
paddingSAME*
strides	
2
functional_15/conv3d_6/Conv3D�
-functional_15/conv3d_6/BiasAdd/ReadVariableOpReadVariableOp6functional_15_conv3d_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-functional_15/conv3d_6/BiasAdd/ReadVariableOp�
functional_15/conv3d_6/BiasAddBiasAdd&functional_15/conv3d_6/Conv3D:output:05functional_15/conv3d_6/BiasAdd/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8������������������������������������2 
functional_15/conv3d_6/BiasAdd�
functional_15/prelu_6/ReluRelu'functional_15/conv3d_6/BiasAdd:output:0*
T0*N
_output_shapes<
::8������������������������������������2
functional_15/prelu_6/Relu�
$functional_15/prelu_6/ReadVariableOpReadVariableOp-functional_15_prelu_6_readvariableop_resource*&
_output_shapes
:*
dtype02&
$functional_15/prelu_6/ReadVariableOp�
functional_15/prelu_6/NegNeg,functional_15/prelu_6/ReadVariableOp:value:0*
T0*&
_output_shapes
:2
functional_15/prelu_6/Neg�
functional_15/prelu_6/Neg_1Neg'functional_15/conv3d_6/BiasAdd:output:0*
T0*N
_output_shapes<
::8������������������������������������2
functional_15/prelu_6/Neg_1�
functional_15/prelu_6/Relu_1Relufunctional_15/prelu_6/Neg_1:y:0*
T0*N
_output_shapes<
::8������������������������������������2
functional_15/prelu_6/Relu_1�
functional_15/prelu_6/mulMulfunctional_15/prelu_6/Neg:y:0*functional_15/prelu_6/Relu_1:activations:0*
T0*N
_output_shapes<
::8������������������������������������2
functional_15/prelu_6/mul�
functional_15/prelu_6/addAddV2(functional_15/prelu_6/Relu:activations:0functional_15/prelu_6/mul:z:0*
T0*N
_output_shapes<
::8������������������������������������2
functional_15/prelu_6/add�
,functional_15/conv_111/Conv3D/ReadVariableOpReadVariableOp5functional_15_conv_111_conv3d_readvariableop_resource**
_output_shapes
:*
dtype02.
,functional_15/conv_111/Conv3D/ReadVariableOp�
functional_15/conv_111/Conv3DConv3Dfunctional_15/prelu_6/add:z:04functional_15/conv_111/Conv3D/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8������������������������������������*
paddingSAME*
strides	
2
functional_15/conv_111/Conv3D�
-functional_15/conv_111/BiasAdd/ReadVariableOpReadVariableOp6functional_15_conv_111_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-functional_15/conv_111/BiasAdd/ReadVariableOp�
functional_15/conv_111/BiasAddBiasAdd&functional_15/conv_111/Conv3D:output:05functional_15/conv_111/BiasAdd/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8������������������������������������2 
functional_15/conv_111/BiasAdd�
functional_15/add_0/addAddV2'functional_15/conv_111/BiasAdd:output:0input_0*
T0*N
_output_shapes<
::8������������������������������������2
functional_15/add_0/add�
IdentityIdentityfunctional_15/add_0/add:z:0*
T0*N
_output_shapes<
::8������������������������������������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:8������������������������������������:8������������������������������������::::::::::::::::::::::::::::::w s
N
_output_shapes<
::8������������������������������������
!
_user_specified_name	input_0:ws
N
_output_shapes<
::8������������������������������������
!
_user_specified_name	input_1
�
l
&__inference_prelu_6_layer_call_fn_4305

inputs
unknown
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8������������������������������������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_prelu_6_layer_call_and_return_conditional_losses_42972
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*N
_output_shapes<
::8������������������������������������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:8������������������������������������:22
StatefulPartitionedCallStatefulPartitionedCall:v r
N
_output_shapes<
::8������������������������������������
 
_user_specified_nameinputs
�
|
'__inference_conv3d_2_layer_call_fn_5592

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8������������������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_conv3d_2_layer_call_and_return_conditional_losses_44522
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*N
_output_shapes<
::8������������������������������������2

Identity"
identityIdentity:output:0*U
_input_shapesD
B:8������������������������������������::22
StatefulPartitionedCallStatefulPartitionedCall:v r
N
_output_shapes<
::8������������������������������������
 
_user_specified_nameinputs
�	
�
B__inference_conv3d_2_layer_call_and_return_conditional_losses_4452

inputs"
conv3d_readvariableop_resource#
biasadd_readvariableop_resource
identity��
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype02
Conv3D/ReadVariableOp�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8������������������������������������*
paddingSAME*
strides	
2
Conv3D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8������������������������������������2	
BiasAdd�
IdentityIdentityBiasAdd:output:0*
T0*N
_output_shapes<
::8������������������������������������2

Identity"
identityIdentity:output:0*U
_input_shapesD
B:8������������������������������������:::v r
N
_output_shapes<
::8������������������������������������
 
_user_specified_nameinputs
�
�
,__inference_functional_15_layer_call_fn_5484
inputs_0
inputs_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8������������������������������������*?
_read_only_resource_inputs!
	
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_functional_15_layer_call_and_return_conditional_losses_49492
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*N
_output_shapes<
::8������������������������������������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:8������������������������������������:8������������������������������������:::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:x t
N
_output_shapes<
::8������������������������������������
"
_user_specified_name
inputs/0:xt
N
_output_shapes<
::8������������������������������������
"
_user_specified_name
inputs/1
�
P
$__inference_add_0_layer_call_fn_5699
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8������������������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_add_0_layer_call_and_return_conditional_losses_46192
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*N
_output_shapes<
::8������������������������������������2

Identity"
identityIdentity:output:0*�
_input_shapesv
t:8������������������������������������:8������������������������������������:x t
N
_output_shapes<
::8������������������������������������
"
_user_specified_name
inputs/0:xt
N
_output_shapes<
::8������������������������������������
"
_user_specified_name
inputs/1
�
l
&__inference_prelu_5_layer_call_fn_4284

inputs
unknown
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8������������������������������������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_prelu_5_layer_call_and_return_conditional_losses_42762
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*N
_output_shapes<
::8������������������������������������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:8������������������������������������:22
StatefulPartitionedCallStatefulPartitionedCall:v r
N
_output_shapes<
::8������������������������������������
 
_user_specified_nameinputs
�

}
A__inference_prelu_5_layer_call_and_return_conditional_losses_4276

inputs
readvariableop_resource
identity�u
ReluReluinputs*
T0*N
_output_shapes<
::8������������������������������������2
Relu�
ReadVariableOpReadVariableOpreadvariableop_resource*&
_output_shapes
:*
dtype02
ReadVariableOpZ
NegNegReadVariableOp:value:0*
T0*&
_output_shapes
:2
Negv
Neg_1Neginputs*
T0*N
_output_shapes<
::8������������������������������������2
Neg_1|
Relu_1Relu	Neg_1:y:0*
T0*N
_output_shapes<
::8������������������������������������2
Relu_1�
mulMulNeg:y:0Relu_1:activations:0*
T0*N
_output_shapes<
::8������������������������������������2
mul�
addAddV2Relu:activations:0mul:z:0*
T0*N
_output_shapes<
::8������������������������������������2
add�
IdentityIdentityadd:z:0*
T0*N
_output_shapes<
::8������������������������������������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:8������������������������������������::v r
N
_output_shapes<
::8������������������������������������
 
_user_specified_nameinputs
�
l
&__inference_prelu_4_layer_call_fn_4263

inputs
unknown
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8������������������������������������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_prelu_4_layer_call_and_return_conditional_losses_42552
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*N
_output_shapes<
::8������������������������������������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:8������������������������������������:22
StatefulPartitionedCallStatefulPartitionedCall:v r
N
_output_shapes<
::8������������������������������������
 
_user_specified_nameinputs
�

}
A__inference_prelu_0_layer_call_and_return_conditional_losses_4171

inputs
readvariableop_resource
identity�u
ReluReluinputs*
T0*N
_output_shapes<
::8������������������������������������2
Relu�
ReadVariableOpReadVariableOpreadvariableop_resource*&
_output_shapes
:*
dtype02
ReadVariableOpZ
NegNegReadVariableOp:value:0*
T0*&
_output_shapes
:2
Negv
Neg_1Neginputs*
T0*N
_output_shapes<
::8������������������������������������2
Neg_1|
Relu_1Relu	Neg_1:y:0*
T0*N
_output_shapes<
::8������������������������������������2
Relu_1�
mulMulNeg:y:0Relu_1:activations:0*
T0*N
_output_shapes<
::8������������������������������������2
mul�
addAddV2Relu:activations:0mul:z:0*
T0*N
_output_shapes<
::8������������������������������������2
add�
IdentityIdentityadd:z:0*
T0*N
_output_shapes<
::8������������������������������������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:8������������������������������������::v r
N
_output_shapes<
::8������������������������������������
 
_user_specified_nameinputs
�
l
B__inference_concat_0_layer_call_and_return_conditional_losses_4375

inputs
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis�
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*N
_output_shapes<
::8������������������������������������2
concat�
IdentityIdentityconcat:output:0*
T0*N
_output_shapes<
::8������������������������������������2

Identity"
identityIdentity:output:0*�
_input_shapesv
t:8������������������������������������:8������������������������������������:v r
N
_output_shapes<
::8������������������������������������
 
_user_specified_nameinputs:vr
N
_output_shapes<
::8������������������������������������
 
_user_specified_nameinputs
�	
�
H__inference_conv3d_ind_0_0_layer_call_and_return_conditional_losses_4346

inputs"
conv3d_readvariableop_resource#
biasadd_readvariableop_resource
identity��
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype02
Conv3D/ReadVariableOp�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8������������������������������������*
paddingSAME*
strides	
2
Conv3D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8������������������������������������2	
BiasAdd�
IdentityIdentityBiasAdd:output:0*
T0*N
_output_shapes<
::8������������������������������������2

Identity"
identityIdentity:output:0*U
_input_shapesD
B:8������������������������������������:::v r
N
_output_shapes<
::8������������������������������������
 
_user_specified_nameinputs
�	
�
B__inference_conv3d_1_layer_call_and_return_conditional_losses_5564

inputs"
conv3d_readvariableop_resource#
biasadd_readvariableop_resource
identity��
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype02
Conv3D/ReadVariableOp�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8������������������������������������*
paddingSAME*
strides	
2
Conv3D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8������������������������������������2	
BiasAdd�
IdentityIdentityBiasAdd:output:0*
T0*N
_output_shapes<
::8������������������������������������2

Identity"
identityIdentity:output:0*U
_input_shapesD
B:8������������������������������������:::v r
N
_output_shapes<
::8������������������������������������
 
_user_specified_nameinputs
�	
�
B__inference_conv3d_2_layer_call_and_return_conditional_losses_5583

inputs"
conv3d_readvariableop_resource#
biasadd_readvariableop_resource
identity��
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype02
Conv3D/ReadVariableOp�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8������������������������������������*
paddingSAME*
strides	
2
Conv3D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8������������������������������������2	
BiasAdd�
IdentityIdentityBiasAdd:output:0*
T0*N
_output_shapes<
::8������������������������������������2

Identity"
identityIdentity:output:0*U
_input_shapesD
B:8������������������������������������:::v r
N
_output_shapes<
::8������������������������������������
 
_user_specified_nameinputs
�
|
'__inference_conv3d_3_layer_call_fn_5611

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8������������������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_conv3d_3_layer_call_and_return_conditional_losses_44812
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*N
_output_shapes<
::8������������������������������������2

Identity"
identityIdentity:output:0*U
_input_shapesD
B:8������������������������������������::22
StatefulPartitionedCallStatefulPartitionedCall:v r
N
_output_shapes<
::8������������������������������������
 
_user_specified_nameinputs
�
�
-__inference_conv3d_ind_0_1_layer_call_fn_5522

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8������������������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_conv3d_ind_0_1_layer_call_and_return_conditional_losses_43202
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*N
_output_shapes<
::8������������������������������������2

Identity"
identityIdentity:output:0*U
_input_shapesD
B:8������������������������������������::22
StatefulPartitionedCallStatefulPartitionedCall:v r
N
_output_shapes<
::8������������������������������������
 
_user_specified_nameinputs
�
l
&__inference_prelu_3_layer_call_fn_4242

inputs
unknown
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8������������������������������������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_prelu_3_layer_call_and_return_conditional_losses_42342
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*N
_output_shapes<
::8������������������������������������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:8������������������������������������:22
StatefulPartitionedCallStatefulPartitionedCall:v r
N
_output_shapes<
::8������������������������������������
 
_user_specified_nameinputs
�	
�
B__inference_conv3d_4_layer_call_and_return_conditional_losses_4510

inputs"
conv3d_readvariableop_resource#
biasadd_readvariableop_resource
identity��
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype02
Conv3D/ReadVariableOp�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8������������������������������������*
paddingSAME*
strides	
2
Conv3D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8������������������������������������2	
BiasAdd�
IdentityIdentityBiasAdd:output:0*
T0*N
_output_shapes<
::8������������������������������������2

Identity"
identityIdentity:output:0*U
_input_shapesD
B:8������������������������������������:::v r
N
_output_shapes<
::8������������������������������������
 
_user_specified_nameinputs
�
|
'__inference_conv3d_5_layer_call_fn_5649

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8������������������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_conv3d_5_layer_call_and_return_conditional_losses_45392
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*N
_output_shapes<
::8������������������������������������2

Identity"
identityIdentity:output:0*U
_input_shapesD
B:8������������������������������������::22
StatefulPartitionedCallStatefulPartitionedCall:v r
N
_output_shapes<
::8������������������������������������
 
_user_specified_nameinputs
�B
�
__inference__traced_save_5810
file_prefix4
0savev2_conv3d_ind_0_0_kernel_read_readvariableop2
.savev2_conv3d_ind_0_0_bias_read_readvariableop4
0savev2_conv3d_ind_0_1_kernel_read_readvariableop2
.savev2_conv3d_ind_0_1_bias_read_readvariableop2
.savev2_prelu_ind_0_0_alpha_read_readvariableop2
.savev2_prelu_ind_0_1_alpha_read_readvariableop.
*savev2_conv3d_0_kernel_read_readvariableop,
(savev2_conv3d_0_bias_read_readvariableop,
(savev2_prelu_0_alpha_read_readvariableop.
*savev2_conv3d_1_kernel_read_readvariableop,
(savev2_conv3d_1_bias_read_readvariableop,
(savev2_prelu_1_alpha_read_readvariableop.
*savev2_conv3d_2_kernel_read_readvariableop,
(savev2_conv3d_2_bias_read_readvariableop,
(savev2_prelu_2_alpha_read_readvariableop.
*savev2_conv3d_3_kernel_read_readvariableop,
(savev2_conv3d_3_bias_read_readvariableop,
(savev2_prelu_3_alpha_read_readvariableop.
*savev2_conv3d_4_kernel_read_readvariableop,
(savev2_conv3d_4_bias_read_readvariableop,
(savev2_prelu_4_alpha_read_readvariableop.
*savev2_conv3d_5_kernel_read_readvariableop,
(savev2_conv3d_5_bias_read_readvariableop,
(savev2_prelu_5_alpha_read_readvariableop.
*savev2_conv3d_6_kernel_read_readvariableop,
(savev2_conv3d_6_bias_read_readvariableop,
(savev2_prelu_6_alpha_read_readvariableop.
*savev2_conv_111_kernel_read_readvariableop,
(savev2_conv_111_bias_read_readvariableop
savev2_const

identity_1��MergeV2Checkpoints�
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const�
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_5cd17a7806d74a3491ab04877111716b/part2	
Const_1�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/alpha/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/alpha/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/alpha/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/alpha/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/alpha/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/alpha/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-13/alpha/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-15/alpha/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-17/alpha/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*O
valueFBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:00savev2_conv3d_ind_0_0_kernel_read_readvariableop.savev2_conv3d_ind_0_0_bias_read_readvariableop0savev2_conv3d_ind_0_1_kernel_read_readvariableop.savev2_conv3d_ind_0_1_bias_read_readvariableop.savev2_prelu_ind_0_0_alpha_read_readvariableop.savev2_prelu_ind_0_1_alpha_read_readvariableop*savev2_conv3d_0_kernel_read_readvariableop(savev2_conv3d_0_bias_read_readvariableop(savev2_prelu_0_alpha_read_readvariableop*savev2_conv3d_1_kernel_read_readvariableop(savev2_conv3d_1_bias_read_readvariableop(savev2_prelu_1_alpha_read_readvariableop*savev2_conv3d_2_kernel_read_readvariableop(savev2_conv3d_2_bias_read_readvariableop(savev2_prelu_2_alpha_read_readvariableop*savev2_conv3d_3_kernel_read_readvariableop(savev2_conv3d_3_bias_read_readvariableop(savev2_prelu_3_alpha_read_readvariableop*savev2_conv3d_4_kernel_read_readvariableop(savev2_conv3d_4_bias_read_readvariableop(savev2_prelu_4_alpha_read_readvariableop*savev2_conv3d_5_kernel_read_readvariableop(savev2_conv3d_5_bias_read_readvariableop(savev2_prelu_5_alpha_read_readvariableop*savev2_conv3d_6_kernel_read_readvariableop(savev2_conv3d_6_bias_read_readvariableop(savev2_prelu_6_alpha_read_readvariableop*savev2_conv_111_kernel_read_readvariableop(savev2_conv_111_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *,
dtypes"
 22
SaveV2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*�
_input_shapes�
�: :::::::::::::::::::::::::::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:0,
*
_output_shapes
:: 

_output_shapes
::0,
*
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
::,(
&
_output_shapes
::0,
*
_output_shapes
:: 

_output_shapes
::,	(
&
_output_shapes
::0
,
*
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
::0,
*
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
::0,
*
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
::0,
*
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
::0,
*
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
::0,
*
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
::0,
*
_output_shapes
:: 

_output_shapes
::

_output_shapes
: 
�l
�

G__inference_functional_15_layer_call_and_return_conditional_losses_4713
input_0
input_1
conv3d_ind_0_1_4633
conv3d_ind_0_1_4635
conv3d_ind_0_0_4638
conv3d_ind_0_0_4640
prelu_ind_0_0_4643
prelu_ind_0_1_4646
conv3d_0_4650
conv3d_0_4652
prelu_0_4655
conv3d_1_4658
conv3d_1_4660
prelu_1_4663
conv3d_2_4666
conv3d_2_4668
prelu_2_4671
conv3d_3_4674
conv3d_3_4676
prelu_3_4679
conv3d_4_4682
conv3d_4_4684
prelu_4_4687
conv3d_5_4690
conv3d_5_4692
prelu_5_4695
conv3d_6_4698
conv3d_6_4700
prelu_6_4703
conv_111_4706
conv_111_4708
identity�� conv3d_0/StatefulPartitionedCall� conv3d_1/StatefulPartitionedCall� conv3d_2/StatefulPartitionedCall� conv3d_3/StatefulPartitionedCall� conv3d_4/StatefulPartitionedCall� conv3d_5/StatefulPartitionedCall� conv3d_6/StatefulPartitionedCall�&conv3d_ind_0_0/StatefulPartitionedCall�&conv3d_ind_0_1/StatefulPartitionedCall� conv_111/StatefulPartitionedCall�prelu_0/StatefulPartitionedCall�prelu_1/StatefulPartitionedCall�prelu_2/StatefulPartitionedCall�prelu_3/StatefulPartitionedCall�prelu_4/StatefulPartitionedCall�prelu_5/StatefulPartitionedCall�prelu_6/StatefulPartitionedCall�%prelu_ind_0_0/StatefulPartitionedCall�%prelu_ind_0_1/StatefulPartitionedCall�
&conv3d_ind_0_1/StatefulPartitionedCallStatefulPartitionedCallinput_1conv3d_ind_0_1_4633conv3d_ind_0_1_4635*
Tin
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8������������������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_conv3d_ind_0_1_layer_call_and_return_conditional_losses_43202(
&conv3d_ind_0_1/StatefulPartitionedCall�
&conv3d_ind_0_0/StatefulPartitionedCallStatefulPartitionedCallinput_0conv3d_ind_0_0_4638conv3d_ind_0_0_4640*
Tin
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8������������������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_conv3d_ind_0_0_layer_call_and_return_conditional_losses_43462(
&conv3d_ind_0_0/StatefulPartitionedCall�
%prelu_ind_0_0/StatefulPartitionedCallStatefulPartitionedCall/conv3d_ind_0_0/StatefulPartitionedCall:output:0prelu_ind_0_0_4643*
Tin
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8������������������������������������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_prelu_ind_0_0_layer_call_and_return_conditional_losses_41292'
%prelu_ind_0_0/StatefulPartitionedCall�
%prelu_ind_0_1/StatefulPartitionedCallStatefulPartitionedCall/conv3d_ind_0_1/StatefulPartitionedCall:output:0prelu_ind_0_1_4646*
Tin
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8������������������������������������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_prelu_ind_0_1_layer_call_and_return_conditional_losses_41502'
%prelu_ind_0_1/StatefulPartitionedCall�
concat_0/PartitionedCallPartitionedCall.prelu_ind_0_0/StatefulPartitionedCall:output:0.prelu_ind_0_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8������������������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_concat_0_layer_call_and_return_conditional_losses_43752
concat_0/PartitionedCall�
 conv3d_0/StatefulPartitionedCallStatefulPartitionedCall!concat_0/PartitionedCall:output:0conv3d_0_4650conv3d_0_4652*
Tin
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8������������������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_conv3d_0_layer_call_and_return_conditional_losses_43942"
 conv3d_0/StatefulPartitionedCall�
prelu_0/StatefulPartitionedCallStatefulPartitionedCall)conv3d_0/StatefulPartitionedCall:output:0prelu_0_4655*
Tin
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8������������������������������������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_prelu_0_layer_call_and_return_conditional_losses_41712!
prelu_0/StatefulPartitionedCall�
 conv3d_1/StatefulPartitionedCallStatefulPartitionedCall(prelu_0/StatefulPartitionedCall:output:0conv3d_1_4658conv3d_1_4660*
Tin
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8������������������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_conv3d_1_layer_call_and_return_conditional_losses_44232"
 conv3d_1/StatefulPartitionedCall�
prelu_1/StatefulPartitionedCallStatefulPartitionedCall)conv3d_1/StatefulPartitionedCall:output:0prelu_1_4663*
Tin
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8������������������������������������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_prelu_1_layer_call_and_return_conditional_losses_41922!
prelu_1/StatefulPartitionedCall�
 conv3d_2/StatefulPartitionedCallStatefulPartitionedCall(prelu_1/StatefulPartitionedCall:output:0conv3d_2_4666conv3d_2_4668*
Tin
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8������������������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_conv3d_2_layer_call_and_return_conditional_losses_44522"
 conv3d_2/StatefulPartitionedCall�
prelu_2/StatefulPartitionedCallStatefulPartitionedCall)conv3d_2/StatefulPartitionedCall:output:0prelu_2_4671*
Tin
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8������������������������������������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_prelu_2_layer_call_and_return_conditional_losses_42132!
prelu_2/StatefulPartitionedCall�
 conv3d_3/StatefulPartitionedCallStatefulPartitionedCall(prelu_2/StatefulPartitionedCall:output:0conv3d_3_4674conv3d_3_4676*
Tin
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8������������������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_conv3d_3_layer_call_and_return_conditional_losses_44812"
 conv3d_3/StatefulPartitionedCall�
prelu_3/StatefulPartitionedCallStatefulPartitionedCall)conv3d_3/StatefulPartitionedCall:output:0prelu_3_4679*
Tin
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8������������������������������������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_prelu_3_layer_call_and_return_conditional_losses_42342!
prelu_3/StatefulPartitionedCall�
 conv3d_4/StatefulPartitionedCallStatefulPartitionedCall(prelu_3/StatefulPartitionedCall:output:0conv3d_4_4682conv3d_4_4684*
Tin
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8������������������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_conv3d_4_layer_call_and_return_conditional_losses_45102"
 conv3d_4/StatefulPartitionedCall�
prelu_4/StatefulPartitionedCallStatefulPartitionedCall)conv3d_4/StatefulPartitionedCall:output:0prelu_4_4687*
Tin
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8������������������������������������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_prelu_4_layer_call_and_return_conditional_losses_42552!
prelu_4/StatefulPartitionedCall�
 conv3d_5/StatefulPartitionedCallStatefulPartitionedCall(prelu_4/StatefulPartitionedCall:output:0conv3d_5_4690conv3d_5_4692*
Tin
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8������������������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_conv3d_5_layer_call_and_return_conditional_losses_45392"
 conv3d_5/StatefulPartitionedCall�
prelu_5/StatefulPartitionedCallStatefulPartitionedCall)conv3d_5/StatefulPartitionedCall:output:0prelu_5_4695*
Tin
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8������������������������������������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_prelu_5_layer_call_and_return_conditional_losses_42762!
prelu_5/StatefulPartitionedCall�
 conv3d_6/StatefulPartitionedCallStatefulPartitionedCall(prelu_5/StatefulPartitionedCall:output:0conv3d_6_4698conv3d_6_4700*
Tin
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8������������������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_conv3d_6_layer_call_and_return_conditional_losses_45682"
 conv3d_6/StatefulPartitionedCall�
prelu_6/StatefulPartitionedCallStatefulPartitionedCall)conv3d_6/StatefulPartitionedCall:output:0prelu_6_4703*
Tin
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8������������������������������������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_prelu_6_layer_call_and_return_conditional_losses_42972!
prelu_6/StatefulPartitionedCall�
 conv_111/StatefulPartitionedCallStatefulPartitionedCall(prelu_6/StatefulPartitionedCall:output:0conv_111_4706conv_111_4708*
Tin
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8������������������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_conv_111_layer_call_and_return_conditional_losses_45972"
 conv_111/StatefulPartitionedCall�
add_0/PartitionedCallPartitionedCall)conv_111/StatefulPartitionedCall:output:0input_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8������������������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_add_0_layer_call_and_return_conditional_losses_46192
add_0/PartitionedCall�
IdentityIdentityadd_0/PartitionedCall:output:0!^conv3d_0/StatefulPartitionedCall!^conv3d_1/StatefulPartitionedCall!^conv3d_2/StatefulPartitionedCall!^conv3d_3/StatefulPartitionedCall!^conv3d_4/StatefulPartitionedCall!^conv3d_5/StatefulPartitionedCall!^conv3d_6/StatefulPartitionedCall'^conv3d_ind_0_0/StatefulPartitionedCall'^conv3d_ind_0_1/StatefulPartitionedCall!^conv_111/StatefulPartitionedCall ^prelu_0/StatefulPartitionedCall ^prelu_1/StatefulPartitionedCall ^prelu_2/StatefulPartitionedCall ^prelu_3/StatefulPartitionedCall ^prelu_4/StatefulPartitionedCall ^prelu_5/StatefulPartitionedCall ^prelu_6/StatefulPartitionedCall&^prelu_ind_0_0/StatefulPartitionedCall&^prelu_ind_0_1/StatefulPartitionedCall*
T0*N
_output_shapes<
::8������������������������������������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:8������������������������������������:8������������������������������������:::::::::::::::::::::::::::::2D
 conv3d_0/StatefulPartitionedCall conv3d_0/StatefulPartitionedCall2D
 conv3d_1/StatefulPartitionedCall conv3d_1/StatefulPartitionedCall2D
 conv3d_2/StatefulPartitionedCall conv3d_2/StatefulPartitionedCall2D
 conv3d_3/StatefulPartitionedCall conv3d_3/StatefulPartitionedCall2D
 conv3d_4/StatefulPartitionedCall conv3d_4/StatefulPartitionedCall2D
 conv3d_5/StatefulPartitionedCall conv3d_5/StatefulPartitionedCall2D
 conv3d_6/StatefulPartitionedCall conv3d_6/StatefulPartitionedCall2P
&conv3d_ind_0_0/StatefulPartitionedCall&conv3d_ind_0_0/StatefulPartitionedCall2P
&conv3d_ind_0_1/StatefulPartitionedCall&conv3d_ind_0_1/StatefulPartitionedCall2D
 conv_111/StatefulPartitionedCall conv_111/StatefulPartitionedCall2B
prelu_0/StatefulPartitionedCallprelu_0/StatefulPartitionedCall2B
prelu_1/StatefulPartitionedCallprelu_1/StatefulPartitionedCall2B
prelu_2/StatefulPartitionedCallprelu_2/StatefulPartitionedCall2B
prelu_3/StatefulPartitionedCallprelu_3/StatefulPartitionedCall2B
prelu_4/StatefulPartitionedCallprelu_4/StatefulPartitionedCall2B
prelu_5/StatefulPartitionedCallprelu_5/StatefulPartitionedCall2B
prelu_6/StatefulPartitionedCallprelu_6/StatefulPartitionedCall2N
%prelu_ind_0_0/StatefulPartitionedCall%prelu_ind_0_0/StatefulPartitionedCall2N
%prelu_ind_0_1/StatefulPartitionedCall%prelu_ind_0_1/StatefulPartitionedCall:w s
N
_output_shapes<
::8������������������������������������
!
_user_specified_name	input_0:ws
N
_output_shapes<
::8������������������������������������
!
_user_specified_name	input_1
�
r
,__inference_prelu_ind_0_1_layer_call_fn_4158

inputs
unknown
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8������������������������������������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_prelu_ind_0_1_layer_call_and_return_conditional_losses_41502
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*N
_output_shapes<
::8������������������������������������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:8������������������������������������:22
StatefulPartitionedCallStatefulPartitionedCall:v r
N
_output_shapes<
::8������������������������������������
 
_user_specified_nameinputs
�

}
A__inference_prelu_6_layer_call_and_return_conditional_losses_4297

inputs
readvariableop_resource
identity�u
ReluReluinputs*
T0*N
_output_shapes<
::8������������������������������������2
Relu�
ReadVariableOpReadVariableOpreadvariableop_resource*&
_output_shapes
:*
dtype02
ReadVariableOpZ
NegNegReadVariableOp:value:0*
T0*&
_output_shapes
:2
Negv
Neg_1Neginputs*
T0*N
_output_shapes<
::8������������������������������������2
Neg_1|
Relu_1Relu	Neg_1:y:0*
T0*N
_output_shapes<
::8������������������������������������2
Relu_1�
mulMulNeg:y:0Relu_1:activations:0*
T0*N
_output_shapes<
::8������������������������������������2
mul�
addAddV2Relu:activations:0mul:z:0*
T0*N
_output_shapes<
::8������������������������������������2
add�
IdentityIdentityadd:z:0*
T0*N
_output_shapes<
::8������������������������������������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:8������������������������������������::v r
N
_output_shapes<
::8������������������������������������
 
_user_specified_nameinputs
�	
�
H__inference_conv3d_ind_0_0_layer_call_and_return_conditional_losses_5494

inputs"
conv3d_readvariableop_resource#
biasadd_readvariableop_resource
identity��
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype02
Conv3D/ReadVariableOp�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8������������������������������������*
paddingSAME*
strides	
2
Conv3D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8������������������������������������2	
BiasAdd�
IdentityIdentityBiasAdd:output:0*
T0*N
_output_shapes<
::8������������������������������������2

Identity"
identityIdentity:output:0*U
_input_shapesD
B:8������������������������������������:::v r
N
_output_shapes<
::8������������������������������������
 
_user_specified_nameinputs
�
�
"__inference_signature_wrapper_5076
input_0
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_0input_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8������������������������������������*?
_read_only_resource_inputs!
	
*-
config_proto

CPU

GPU 2J 8� *(
f#R!
__inference__wrapped_model_41162
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*N
_output_shapes<
::8������������������������������������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:8������������������������������������:8������������������������������������:::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:w s
N
_output_shapes<
::8������������������������������������
!
_user_specified_name	input_0:ws
N
_output_shapes<
::8������������������������������������
!
_user_specified_name	input_1
�
|
'__inference_conv3d_1_layer_call_fn_5573

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8������������������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_conv3d_1_layer_call_and_return_conditional_losses_44232
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*N
_output_shapes<
::8������������������������������������2

Identity"
identityIdentity:output:0*U
_input_shapesD
B:8������������������������������������::22
StatefulPartitionedCallStatefulPartitionedCall:v r
N
_output_shapes<
::8������������������������������������
 
_user_specified_nameinputs
�

}
A__inference_prelu_3_layer_call_and_return_conditional_losses_4234

inputs
readvariableop_resource
identity�u
ReluReluinputs*
T0*N
_output_shapes<
::8������������������������������������2
Relu�
ReadVariableOpReadVariableOpreadvariableop_resource*&
_output_shapes
:*
dtype02
ReadVariableOpZ
NegNegReadVariableOp:value:0*
T0*&
_output_shapes
:2
Negv
Neg_1Neginputs*
T0*N
_output_shapes<
::8������������������������������������2
Neg_1|
Relu_1Relu	Neg_1:y:0*
T0*N
_output_shapes<
::8������������������������������������2
Relu_1�
mulMulNeg:y:0Relu_1:activations:0*
T0*N
_output_shapes<
::8������������������������������������2
mul�
addAddV2Relu:activations:0mul:z:0*
T0*N
_output_shapes<
::8������������������������������������2
add�
IdentityIdentityadd:z:0*
T0*N
_output_shapes<
::8������������������������������������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:8������������������������������������::v r
N
_output_shapes<
::8������������������������������������
 
_user_specified_nameinputs
�l
�

G__inference_functional_15_layer_call_and_return_conditional_losses_4949

inputs
inputs_1
conv3d_ind_0_1_4869
conv3d_ind_0_1_4871
conv3d_ind_0_0_4874
conv3d_ind_0_0_4876
prelu_ind_0_0_4879
prelu_ind_0_1_4882
conv3d_0_4886
conv3d_0_4888
prelu_0_4891
conv3d_1_4894
conv3d_1_4896
prelu_1_4899
conv3d_2_4902
conv3d_2_4904
prelu_2_4907
conv3d_3_4910
conv3d_3_4912
prelu_3_4915
conv3d_4_4918
conv3d_4_4920
prelu_4_4923
conv3d_5_4926
conv3d_5_4928
prelu_5_4931
conv3d_6_4934
conv3d_6_4936
prelu_6_4939
conv_111_4942
conv_111_4944
identity�� conv3d_0/StatefulPartitionedCall� conv3d_1/StatefulPartitionedCall� conv3d_2/StatefulPartitionedCall� conv3d_3/StatefulPartitionedCall� conv3d_4/StatefulPartitionedCall� conv3d_5/StatefulPartitionedCall� conv3d_6/StatefulPartitionedCall�&conv3d_ind_0_0/StatefulPartitionedCall�&conv3d_ind_0_1/StatefulPartitionedCall� conv_111/StatefulPartitionedCall�prelu_0/StatefulPartitionedCall�prelu_1/StatefulPartitionedCall�prelu_2/StatefulPartitionedCall�prelu_3/StatefulPartitionedCall�prelu_4/StatefulPartitionedCall�prelu_5/StatefulPartitionedCall�prelu_6/StatefulPartitionedCall�%prelu_ind_0_0/StatefulPartitionedCall�%prelu_ind_0_1/StatefulPartitionedCall�
&conv3d_ind_0_1/StatefulPartitionedCallStatefulPartitionedCallinputs_1conv3d_ind_0_1_4869conv3d_ind_0_1_4871*
Tin
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8������������������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_conv3d_ind_0_1_layer_call_and_return_conditional_losses_43202(
&conv3d_ind_0_1/StatefulPartitionedCall�
&conv3d_ind_0_0/StatefulPartitionedCallStatefulPartitionedCallinputsconv3d_ind_0_0_4874conv3d_ind_0_0_4876*
Tin
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8������������������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_conv3d_ind_0_0_layer_call_and_return_conditional_losses_43462(
&conv3d_ind_0_0/StatefulPartitionedCall�
%prelu_ind_0_0/StatefulPartitionedCallStatefulPartitionedCall/conv3d_ind_0_0/StatefulPartitionedCall:output:0prelu_ind_0_0_4879*
Tin
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8������������������������������������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_prelu_ind_0_0_layer_call_and_return_conditional_losses_41292'
%prelu_ind_0_0/StatefulPartitionedCall�
%prelu_ind_0_1/StatefulPartitionedCallStatefulPartitionedCall/conv3d_ind_0_1/StatefulPartitionedCall:output:0prelu_ind_0_1_4882*
Tin
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8������������������������������������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_prelu_ind_0_1_layer_call_and_return_conditional_losses_41502'
%prelu_ind_0_1/StatefulPartitionedCall�
concat_0/PartitionedCallPartitionedCall.prelu_ind_0_0/StatefulPartitionedCall:output:0.prelu_ind_0_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8������������������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_concat_0_layer_call_and_return_conditional_losses_43752
concat_0/PartitionedCall�
 conv3d_0/StatefulPartitionedCallStatefulPartitionedCall!concat_0/PartitionedCall:output:0conv3d_0_4886conv3d_0_4888*
Tin
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8������������������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_conv3d_0_layer_call_and_return_conditional_losses_43942"
 conv3d_0/StatefulPartitionedCall�
prelu_0/StatefulPartitionedCallStatefulPartitionedCall)conv3d_0/StatefulPartitionedCall:output:0prelu_0_4891*
Tin
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8������������������������������������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_prelu_0_layer_call_and_return_conditional_losses_41712!
prelu_0/StatefulPartitionedCall�
 conv3d_1/StatefulPartitionedCallStatefulPartitionedCall(prelu_0/StatefulPartitionedCall:output:0conv3d_1_4894conv3d_1_4896*
Tin
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8������������������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_conv3d_1_layer_call_and_return_conditional_losses_44232"
 conv3d_1/StatefulPartitionedCall�
prelu_1/StatefulPartitionedCallStatefulPartitionedCall)conv3d_1/StatefulPartitionedCall:output:0prelu_1_4899*
Tin
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8������������������������������������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_prelu_1_layer_call_and_return_conditional_losses_41922!
prelu_1/StatefulPartitionedCall�
 conv3d_2/StatefulPartitionedCallStatefulPartitionedCall(prelu_1/StatefulPartitionedCall:output:0conv3d_2_4902conv3d_2_4904*
Tin
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8������������������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_conv3d_2_layer_call_and_return_conditional_losses_44522"
 conv3d_2/StatefulPartitionedCall�
prelu_2/StatefulPartitionedCallStatefulPartitionedCall)conv3d_2/StatefulPartitionedCall:output:0prelu_2_4907*
Tin
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8������������������������������������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_prelu_2_layer_call_and_return_conditional_losses_42132!
prelu_2/StatefulPartitionedCall�
 conv3d_3/StatefulPartitionedCallStatefulPartitionedCall(prelu_2/StatefulPartitionedCall:output:0conv3d_3_4910conv3d_3_4912*
Tin
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8������������������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_conv3d_3_layer_call_and_return_conditional_losses_44812"
 conv3d_3/StatefulPartitionedCall�
prelu_3/StatefulPartitionedCallStatefulPartitionedCall)conv3d_3/StatefulPartitionedCall:output:0prelu_3_4915*
Tin
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8������������������������������������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_prelu_3_layer_call_and_return_conditional_losses_42342!
prelu_3/StatefulPartitionedCall�
 conv3d_4/StatefulPartitionedCallStatefulPartitionedCall(prelu_3/StatefulPartitionedCall:output:0conv3d_4_4918conv3d_4_4920*
Tin
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8������������������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_conv3d_4_layer_call_and_return_conditional_losses_45102"
 conv3d_4/StatefulPartitionedCall�
prelu_4/StatefulPartitionedCallStatefulPartitionedCall)conv3d_4/StatefulPartitionedCall:output:0prelu_4_4923*
Tin
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8������������������������������������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_prelu_4_layer_call_and_return_conditional_losses_42552!
prelu_4/StatefulPartitionedCall�
 conv3d_5/StatefulPartitionedCallStatefulPartitionedCall(prelu_4/StatefulPartitionedCall:output:0conv3d_5_4926conv3d_5_4928*
Tin
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8������������������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_conv3d_5_layer_call_and_return_conditional_losses_45392"
 conv3d_5/StatefulPartitionedCall�
prelu_5/StatefulPartitionedCallStatefulPartitionedCall)conv3d_5/StatefulPartitionedCall:output:0prelu_5_4931*
Tin
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8������������������������������������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_prelu_5_layer_call_and_return_conditional_losses_42762!
prelu_5/StatefulPartitionedCall�
 conv3d_6/StatefulPartitionedCallStatefulPartitionedCall(prelu_5/StatefulPartitionedCall:output:0conv3d_6_4934conv3d_6_4936*
Tin
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8������������������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_conv3d_6_layer_call_and_return_conditional_losses_45682"
 conv3d_6/StatefulPartitionedCall�
prelu_6/StatefulPartitionedCallStatefulPartitionedCall)conv3d_6/StatefulPartitionedCall:output:0prelu_6_4939*
Tin
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8������������������������������������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_prelu_6_layer_call_and_return_conditional_losses_42972!
prelu_6/StatefulPartitionedCall�
 conv_111/StatefulPartitionedCallStatefulPartitionedCall(prelu_6/StatefulPartitionedCall:output:0conv_111_4942conv_111_4944*
Tin
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8������������������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_conv_111_layer_call_and_return_conditional_losses_45972"
 conv_111/StatefulPartitionedCall�
add_0/PartitionedCallPartitionedCall)conv_111/StatefulPartitionedCall:output:0inputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8������������������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_add_0_layer_call_and_return_conditional_losses_46192
add_0/PartitionedCall�
IdentityIdentityadd_0/PartitionedCall:output:0!^conv3d_0/StatefulPartitionedCall!^conv3d_1/StatefulPartitionedCall!^conv3d_2/StatefulPartitionedCall!^conv3d_3/StatefulPartitionedCall!^conv3d_4/StatefulPartitionedCall!^conv3d_5/StatefulPartitionedCall!^conv3d_6/StatefulPartitionedCall'^conv3d_ind_0_0/StatefulPartitionedCall'^conv3d_ind_0_1/StatefulPartitionedCall!^conv_111/StatefulPartitionedCall ^prelu_0/StatefulPartitionedCall ^prelu_1/StatefulPartitionedCall ^prelu_2/StatefulPartitionedCall ^prelu_3/StatefulPartitionedCall ^prelu_4/StatefulPartitionedCall ^prelu_5/StatefulPartitionedCall ^prelu_6/StatefulPartitionedCall&^prelu_ind_0_0/StatefulPartitionedCall&^prelu_ind_0_1/StatefulPartitionedCall*
T0*N
_output_shapes<
::8������������������������������������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:8������������������������������������:8������������������������������������:::::::::::::::::::::::::::::2D
 conv3d_0/StatefulPartitionedCall conv3d_0/StatefulPartitionedCall2D
 conv3d_1/StatefulPartitionedCall conv3d_1/StatefulPartitionedCall2D
 conv3d_2/StatefulPartitionedCall conv3d_2/StatefulPartitionedCall2D
 conv3d_3/StatefulPartitionedCall conv3d_3/StatefulPartitionedCall2D
 conv3d_4/StatefulPartitionedCall conv3d_4/StatefulPartitionedCall2D
 conv3d_5/StatefulPartitionedCall conv3d_5/StatefulPartitionedCall2D
 conv3d_6/StatefulPartitionedCall conv3d_6/StatefulPartitionedCall2P
&conv3d_ind_0_0/StatefulPartitionedCall&conv3d_ind_0_0/StatefulPartitionedCall2P
&conv3d_ind_0_1/StatefulPartitionedCall&conv3d_ind_0_1/StatefulPartitionedCall2D
 conv_111/StatefulPartitionedCall conv_111/StatefulPartitionedCall2B
prelu_0/StatefulPartitionedCallprelu_0/StatefulPartitionedCall2B
prelu_1/StatefulPartitionedCallprelu_1/StatefulPartitionedCall2B
prelu_2/StatefulPartitionedCallprelu_2/StatefulPartitionedCall2B
prelu_3/StatefulPartitionedCallprelu_3/StatefulPartitionedCall2B
prelu_4/StatefulPartitionedCallprelu_4/StatefulPartitionedCall2B
prelu_5/StatefulPartitionedCallprelu_5/StatefulPartitionedCall2B
prelu_6/StatefulPartitionedCallprelu_6/StatefulPartitionedCall2N
%prelu_ind_0_0/StatefulPartitionedCall%prelu_ind_0_0/StatefulPartitionedCall2N
%prelu_ind_0_1/StatefulPartitionedCall%prelu_ind_0_1/StatefulPartitionedCall:v r
N
_output_shapes<
::8������������������������������������
 
_user_specified_nameinputs:vr
N
_output_shapes<
::8������������������������������������
 
_user_specified_nameinputs
�
|
'__inference_conv3d_4_layer_call_fn_5630

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8������������������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_conv3d_4_layer_call_and_return_conditional_losses_45102
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*N
_output_shapes<
::8������������������������������������2

Identity"
identityIdentity:output:0*U
_input_shapesD
B:8������������������������������������::22
StatefulPartitionedCallStatefulPartitionedCall:v r
N
_output_shapes<
::8������������������������������������
 
_user_specified_nameinputs
�l
�

G__inference_functional_15_layer_call_and_return_conditional_losses_4629
input_0
input_1
conv3d_ind_0_1_4331
conv3d_ind_0_1_4333
conv3d_ind_0_0_4357
conv3d_ind_0_0_4359
prelu_ind_0_0_4362
prelu_ind_0_1_4365
conv3d_0_4405
conv3d_0_4407
prelu_0_4410
conv3d_1_4434
conv3d_1_4436
prelu_1_4439
conv3d_2_4463
conv3d_2_4465
prelu_2_4468
conv3d_3_4492
conv3d_3_4494
prelu_3_4497
conv3d_4_4521
conv3d_4_4523
prelu_4_4526
conv3d_5_4550
conv3d_5_4552
prelu_5_4555
conv3d_6_4579
conv3d_6_4581
prelu_6_4584
conv_111_4608
conv_111_4610
identity�� conv3d_0/StatefulPartitionedCall� conv3d_1/StatefulPartitionedCall� conv3d_2/StatefulPartitionedCall� conv3d_3/StatefulPartitionedCall� conv3d_4/StatefulPartitionedCall� conv3d_5/StatefulPartitionedCall� conv3d_6/StatefulPartitionedCall�&conv3d_ind_0_0/StatefulPartitionedCall�&conv3d_ind_0_1/StatefulPartitionedCall� conv_111/StatefulPartitionedCall�prelu_0/StatefulPartitionedCall�prelu_1/StatefulPartitionedCall�prelu_2/StatefulPartitionedCall�prelu_3/StatefulPartitionedCall�prelu_4/StatefulPartitionedCall�prelu_5/StatefulPartitionedCall�prelu_6/StatefulPartitionedCall�%prelu_ind_0_0/StatefulPartitionedCall�%prelu_ind_0_1/StatefulPartitionedCall�
&conv3d_ind_0_1/StatefulPartitionedCallStatefulPartitionedCallinput_1conv3d_ind_0_1_4331conv3d_ind_0_1_4333*
Tin
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8������������������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_conv3d_ind_0_1_layer_call_and_return_conditional_losses_43202(
&conv3d_ind_0_1/StatefulPartitionedCall�
&conv3d_ind_0_0/StatefulPartitionedCallStatefulPartitionedCallinput_0conv3d_ind_0_0_4357conv3d_ind_0_0_4359*
Tin
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8������������������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_conv3d_ind_0_0_layer_call_and_return_conditional_losses_43462(
&conv3d_ind_0_0/StatefulPartitionedCall�
%prelu_ind_0_0/StatefulPartitionedCallStatefulPartitionedCall/conv3d_ind_0_0/StatefulPartitionedCall:output:0prelu_ind_0_0_4362*
Tin
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8������������������������������������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_prelu_ind_0_0_layer_call_and_return_conditional_losses_41292'
%prelu_ind_0_0/StatefulPartitionedCall�
%prelu_ind_0_1/StatefulPartitionedCallStatefulPartitionedCall/conv3d_ind_0_1/StatefulPartitionedCall:output:0prelu_ind_0_1_4365*
Tin
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8������������������������������������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_prelu_ind_0_1_layer_call_and_return_conditional_losses_41502'
%prelu_ind_0_1/StatefulPartitionedCall�
concat_0/PartitionedCallPartitionedCall.prelu_ind_0_0/StatefulPartitionedCall:output:0.prelu_ind_0_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8������������������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_concat_0_layer_call_and_return_conditional_losses_43752
concat_0/PartitionedCall�
 conv3d_0/StatefulPartitionedCallStatefulPartitionedCall!concat_0/PartitionedCall:output:0conv3d_0_4405conv3d_0_4407*
Tin
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8������������������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_conv3d_0_layer_call_and_return_conditional_losses_43942"
 conv3d_0/StatefulPartitionedCall�
prelu_0/StatefulPartitionedCallStatefulPartitionedCall)conv3d_0/StatefulPartitionedCall:output:0prelu_0_4410*
Tin
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8������������������������������������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_prelu_0_layer_call_and_return_conditional_losses_41712!
prelu_0/StatefulPartitionedCall�
 conv3d_1/StatefulPartitionedCallStatefulPartitionedCall(prelu_0/StatefulPartitionedCall:output:0conv3d_1_4434conv3d_1_4436*
Tin
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8������������������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_conv3d_1_layer_call_and_return_conditional_losses_44232"
 conv3d_1/StatefulPartitionedCall�
prelu_1/StatefulPartitionedCallStatefulPartitionedCall)conv3d_1/StatefulPartitionedCall:output:0prelu_1_4439*
Tin
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8������������������������������������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_prelu_1_layer_call_and_return_conditional_losses_41922!
prelu_1/StatefulPartitionedCall�
 conv3d_2/StatefulPartitionedCallStatefulPartitionedCall(prelu_1/StatefulPartitionedCall:output:0conv3d_2_4463conv3d_2_4465*
Tin
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8������������������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_conv3d_2_layer_call_and_return_conditional_losses_44522"
 conv3d_2/StatefulPartitionedCall�
prelu_2/StatefulPartitionedCallStatefulPartitionedCall)conv3d_2/StatefulPartitionedCall:output:0prelu_2_4468*
Tin
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8������������������������������������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_prelu_2_layer_call_and_return_conditional_losses_42132!
prelu_2/StatefulPartitionedCall�
 conv3d_3/StatefulPartitionedCallStatefulPartitionedCall(prelu_2/StatefulPartitionedCall:output:0conv3d_3_4492conv3d_3_4494*
Tin
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8������������������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_conv3d_3_layer_call_and_return_conditional_losses_44812"
 conv3d_3/StatefulPartitionedCall�
prelu_3/StatefulPartitionedCallStatefulPartitionedCall)conv3d_3/StatefulPartitionedCall:output:0prelu_3_4497*
Tin
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8������������������������������������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_prelu_3_layer_call_and_return_conditional_losses_42342!
prelu_3/StatefulPartitionedCall�
 conv3d_4/StatefulPartitionedCallStatefulPartitionedCall(prelu_3/StatefulPartitionedCall:output:0conv3d_4_4521conv3d_4_4523*
Tin
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8������������������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_conv3d_4_layer_call_and_return_conditional_losses_45102"
 conv3d_4/StatefulPartitionedCall�
prelu_4/StatefulPartitionedCallStatefulPartitionedCall)conv3d_4/StatefulPartitionedCall:output:0prelu_4_4526*
Tin
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8������������������������������������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_prelu_4_layer_call_and_return_conditional_losses_42552!
prelu_4/StatefulPartitionedCall�
 conv3d_5/StatefulPartitionedCallStatefulPartitionedCall(prelu_4/StatefulPartitionedCall:output:0conv3d_5_4550conv3d_5_4552*
Tin
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8������������������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_conv3d_5_layer_call_and_return_conditional_losses_45392"
 conv3d_5/StatefulPartitionedCall�
prelu_5/StatefulPartitionedCallStatefulPartitionedCall)conv3d_5/StatefulPartitionedCall:output:0prelu_5_4555*
Tin
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8������������������������������������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_prelu_5_layer_call_and_return_conditional_losses_42762!
prelu_5/StatefulPartitionedCall�
 conv3d_6/StatefulPartitionedCallStatefulPartitionedCall(prelu_5/StatefulPartitionedCall:output:0conv3d_6_4579conv3d_6_4581*
Tin
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8������������������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_conv3d_6_layer_call_and_return_conditional_losses_45682"
 conv3d_6/StatefulPartitionedCall�
prelu_6/StatefulPartitionedCallStatefulPartitionedCall)conv3d_6/StatefulPartitionedCall:output:0prelu_6_4584*
Tin
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8������������������������������������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_prelu_6_layer_call_and_return_conditional_losses_42972!
prelu_6/StatefulPartitionedCall�
 conv_111/StatefulPartitionedCallStatefulPartitionedCall(prelu_6/StatefulPartitionedCall:output:0conv_111_4608conv_111_4610*
Tin
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8������������������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_conv_111_layer_call_and_return_conditional_losses_45972"
 conv_111/StatefulPartitionedCall�
add_0/PartitionedCallPartitionedCall)conv_111/StatefulPartitionedCall:output:0input_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8������������������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_add_0_layer_call_and_return_conditional_losses_46192
add_0/PartitionedCall�
IdentityIdentityadd_0/PartitionedCall:output:0!^conv3d_0/StatefulPartitionedCall!^conv3d_1/StatefulPartitionedCall!^conv3d_2/StatefulPartitionedCall!^conv3d_3/StatefulPartitionedCall!^conv3d_4/StatefulPartitionedCall!^conv3d_5/StatefulPartitionedCall!^conv3d_6/StatefulPartitionedCall'^conv3d_ind_0_0/StatefulPartitionedCall'^conv3d_ind_0_1/StatefulPartitionedCall!^conv_111/StatefulPartitionedCall ^prelu_0/StatefulPartitionedCall ^prelu_1/StatefulPartitionedCall ^prelu_2/StatefulPartitionedCall ^prelu_3/StatefulPartitionedCall ^prelu_4/StatefulPartitionedCall ^prelu_5/StatefulPartitionedCall ^prelu_6/StatefulPartitionedCall&^prelu_ind_0_0/StatefulPartitionedCall&^prelu_ind_0_1/StatefulPartitionedCall*
T0*N
_output_shapes<
::8������������������������������������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:8������������������������������������:8������������������������������������:::::::::::::::::::::::::::::2D
 conv3d_0/StatefulPartitionedCall conv3d_0/StatefulPartitionedCall2D
 conv3d_1/StatefulPartitionedCall conv3d_1/StatefulPartitionedCall2D
 conv3d_2/StatefulPartitionedCall conv3d_2/StatefulPartitionedCall2D
 conv3d_3/StatefulPartitionedCall conv3d_3/StatefulPartitionedCall2D
 conv3d_4/StatefulPartitionedCall conv3d_4/StatefulPartitionedCall2D
 conv3d_5/StatefulPartitionedCall conv3d_5/StatefulPartitionedCall2D
 conv3d_6/StatefulPartitionedCall conv3d_6/StatefulPartitionedCall2P
&conv3d_ind_0_0/StatefulPartitionedCall&conv3d_ind_0_0/StatefulPartitionedCall2P
&conv3d_ind_0_1/StatefulPartitionedCall&conv3d_ind_0_1/StatefulPartitionedCall2D
 conv_111/StatefulPartitionedCall conv_111/StatefulPartitionedCall2B
prelu_0/StatefulPartitionedCallprelu_0/StatefulPartitionedCall2B
prelu_1/StatefulPartitionedCallprelu_1/StatefulPartitionedCall2B
prelu_2/StatefulPartitionedCallprelu_2/StatefulPartitionedCall2B
prelu_3/StatefulPartitionedCallprelu_3/StatefulPartitionedCall2B
prelu_4/StatefulPartitionedCallprelu_4/StatefulPartitionedCall2B
prelu_5/StatefulPartitionedCallprelu_5/StatefulPartitionedCall2B
prelu_6/StatefulPartitionedCallprelu_6/StatefulPartitionedCall2N
%prelu_ind_0_0/StatefulPartitionedCall%prelu_ind_0_0/StatefulPartitionedCall2N
%prelu_ind_0_1/StatefulPartitionedCall%prelu_ind_0_1/StatefulPartitionedCall:w s
N
_output_shapes<
::8������������������������������������
!
_user_specified_name	input_0:ws
N
_output_shapes<
::8������������������������������������
!
_user_specified_name	input_1
�

}
A__inference_prelu_1_layer_call_and_return_conditional_losses_4192

inputs
readvariableop_resource
identity�u
ReluReluinputs*
T0*N
_output_shapes<
::8������������������������������������2
Relu�
ReadVariableOpReadVariableOpreadvariableop_resource*&
_output_shapes
:*
dtype02
ReadVariableOpZ
NegNegReadVariableOp:value:0*
T0*&
_output_shapes
:2
Negv
Neg_1Neginputs*
T0*N
_output_shapes<
::8������������������������������������2
Neg_1|
Relu_1Relu	Neg_1:y:0*
T0*N
_output_shapes<
::8������������������������������������2
Relu_1�
mulMulNeg:y:0Relu_1:activations:0*
T0*N
_output_shapes<
::8������������������������������������2
mul�
addAddV2Relu:activations:0mul:z:0*
T0*N
_output_shapes<
::8������������������������������������2
add�
IdentityIdentityadd:z:0*
T0*N
_output_shapes<
::8������������������������������������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:8������������������������������������::v r
N
_output_shapes<
::8������������������������������������
 
_user_specified_nameinputs
�	
�
H__inference_conv3d_ind_0_1_layer_call_and_return_conditional_losses_4320

inputs"
conv3d_readvariableop_resource#
biasadd_readvariableop_resource
identity��
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype02
Conv3D/ReadVariableOp�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8������������������������������������*
paddingSAME*
strides	
2
Conv3D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8������������������������������������2	
BiasAdd�
IdentityIdentityBiasAdd:output:0*
T0*N
_output_shapes<
::8������������������������������������2

Identity"
identityIdentity:output:0*U
_input_shapesD
B:8������������������������������������:::v r
N
_output_shapes<
::8������������������������������������
 
_user_specified_nameinputs
�
i
?__inference_add_0_layer_call_and_return_conditional_losses_4619

inputs
inputs_1
identity~
addAddV2inputsinputs_1*
T0*N
_output_shapes<
::8������������������������������������2
add�
IdentityIdentityadd:z:0*
T0*N
_output_shapes<
::8������������������������������������2

Identity"
identityIdentity:output:0*�
_input_shapesv
t:8������������������������������������:8������������������������������������:v r
N
_output_shapes<
::8������������������������������������
 
_user_specified_nameinputs:vr
N
_output_shapes<
::8������������������������������������
 
_user_specified_nameinputs
�	
�
B__inference_conv3d_1_layer_call_and_return_conditional_losses_4423

inputs"
conv3d_readvariableop_resource#
biasadd_readvariableop_resource
identity��
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype02
Conv3D/ReadVariableOp�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8������������������������������������*
paddingSAME*
strides	
2
Conv3D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8������������������������������������2	
BiasAdd�
IdentityIdentityBiasAdd:output:0*
T0*N
_output_shapes<
::8������������������������������������2

Identity"
identityIdentity:output:0*U
_input_shapesD
B:8������������������������������������:::v r
N
_output_shapes<
::8������������������������������������
 
_user_specified_nameinputs
�	
�
B__inference_conv3d_3_layer_call_and_return_conditional_losses_4481

inputs"
conv3d_readvariableop_resource#
biasadd_readvariableop_resource
identity��
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype02
Conv3D/ReadVariableOp�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8������������������������������������*
paddingSAME*
strides	
2
Conv3D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8������������������������������������2	
BiasAdd�
IdentityIdentityBiasAdd:output:0*
T0*N
_output_shapes<
::8������������������������������������2

Identity"
identityIdentity:output:0*U
_input_shapesD
B:8������������������������������������:::v r
N
_output_shapes<
::8������������������������������������
 
_user_specified_nameinputs
�
l
&__inference_prelu_2_layer_call_fn_4221

inputs
unknown
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8������������������������������������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_prelu_2_layer_call_and_return_conditional_losses_42132
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*N
_output_shapes<
::8������������������������������������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:8������������������������������������:22
StatefulPartitionedCallStatefulPartitionedCall:v r
N
_output_shapes<
::8������������������������������������
 
_user_specified_nameinputs
�	
�
B__inference_conv3d_0_layer_call_and_return_conditional_losses_5545

inputs"
conv3d_readvariableop_resource#
biasadd_readvariableop_resource
identity��
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype02
Conv3D/ReadVariableOp�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8������������������������������������*
paddingSAME*
strides	
2
Conv3D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8������������������������������������2	
BiasAdd�
IdentityIdentityBiasAdd:output:0*
T0*N
_output_shapes<
::8������������������������������������2

Identity"
identityIdentity:output:0*U
_input_shapesD
B:8������������������������������������:::v r
N
_output_shapes<
::8������������������������������������
 
_user_specified_nameinputs
�

�
G__inference_prelu_ind_0_0_layer_call_and_return_conditional_losses_4129

inputs
readvariableop_resource
identity�u
ReluReluinputs*
T0*N
_output_shapes<
::8������������������������������������2
Relu�
ReadVariableOpReadVariableOpreadvariableop_resource*&
_output_shapes
:*
dtype02
ReadVariableOpZ
NegNegReadVariableOp:value:0*
T0*&
_output_shapes
:2
Negv
Neg_1Neginputs*
T0*N
_output_shapes<
::8������������������������������������2
Neg_1|
Relu_1Relu	Neg_1:y:0*
T0*N
_output_shapes<
::8������������������������������������2
Relu_1�
mulMulNeg:y:0Relu_1:activations:0*
T0*N
_output_shapes<
::8������������������������������������2
mul�
addAddV2Relu:activations:0mul:z:0*
T0*N
_output_shapes<
::8������������������������������������2
add�
IdentityIdentityadd:z:0*
T0*N
_output_shapes<
::8������������������������������������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:8������������������������������������::v r
N
_output_shapes<
::8������������������������������������
 
_user_specified_nameinputs
�	
�
B__inference_conv3d_4_layer_call_and_return_conditional_losses_5621

inputs"
conv3d_readvariableop_resource#
biasadd_readvariableop_resource
identity��
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype02
Conv3D/ReadVariableOp�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8������������������������������������*
paddingSAME*
strides	
2
Conv3D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8������������������������������������2	
BiasAdd�
IdentityIdentityBiasAdd:output:0*
T0*N
_output_shapes<
::8������������������������������������2

Identity"
identityIdentity:output:0*U
_input_shapesD
B:8������������������������������������:::v r
N
_output_shapes<
::8������������������������������������
 
_user_specified_nameinputs
�
�
,__inference_functional_15_layer_call_fn_5420
inputs_0
inputs_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8������������������������������������*?
_read_only_resource_inputs!
	
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_functional_15_layer_call_and_return_conditional_losses_48012
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*N
_output_shapes<
::8������������������������������������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:8������������������������������������:8������������������������������������:::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:x t
N
_output_shapes<
::8������������������������������������
"
_user_specified_name
inputs/0:xt
N
_output_shapes<
::8������������������������������������
"
_user_specified_name
inputs/1
�
�
-__inference_conv3d_ind_0_0_layer_call_fn_5503

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8������������������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_conv3d_ind_0_0_layer_call_and_return_conditional_losses_43462
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*N
_output_shapes<
::8������������������������������������2

Identity"
identityIdentity:output:0*U
_input_shapesD
B:8������������������������������������::22
StatefulPartitionedCallStatefulPartitionedCall:v r
N
_output_shapes<
::8������������������������������������
 
_user_specified_nameinputs
�
|
'__inference_conv3d_6_layer_call_fn_5668

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8������������������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_conv3d_6_layer_call_and_return_conditional_losses_45682
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*N
_output_shapes<
::8������������������������������������2

Identity"
identityIdentity:output:0*U
_input_shapesD
B:8������������������������������������::22
StatefulPartitionedCallStatefulPartitionedCall:v r
N
_output_shapes<
::8������������������������������������
 
_user_specified_nameinputs
�

}
A__inference_prelu_4_layer_call_and_return_conditional_losses_4255

inputs
readvariableop_resource
identity�u
ReluReluinputs*
T0*N
_output_shapes<
::8������������������������������������2
Relu�
ReadVariableOpReadVariableOpreadvariableop_resource*&
_output_shapes
:*
dtype02
ReadVariableOpZ
NegNegReadVariableOp:value:0*
T0*&
_output_shapes
:2
Negv
Neg_1Neginputs*
T0*N
_output_shapes<
::8������������������������������������2
Neg_1|
Relu_1Relu	Neg_1:y:0*
T0*N
_output_shapes<
::8������������������������������������2
Relu_1�
mulMulNeg:y:0Relu_1:activations:0*
T0*N
_output_shapes<
::8������������������������������������2
mul�
addAddV2Relu:activations:0mul:z:0*
T0*N
_output_shapes<
::8������������������������������������2
add�
IdentityIdentityadd:z:0*
T0*N
_output_shapes<
::8������������������������������������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:8������������������������������������::v r
N
_output_shapes<
::8������������������������������������
 
_user_specified_nameinputs
�
�
,__inference_functional_15_layer_call_fn_4862
input_0
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_0input_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8������������������������������������*?
_read_only_resource_inputs!
	
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_functional_15_layer_call_and_return_conditional_losses_48012
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*N
_output_shapes<
::8������������������������������������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:8������������������������������������:8������������������������������������:::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:w s
N
_output_shapes<
::8������������������������������������
!
_user_specified_name	input_0:ws
N
_output_shapes<
::8������������������������������������
!
_user_specified_name	input_1
�
r
,__inference_prelu_ind_0_0_layer_call_fn_4137

inputs
unknown
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8������������������������������������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_prelu_ind_0_0_layer_call_and_return_conditional_losses_41292
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*N
_output_shapes<
::8������������������������������������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:8������������������������������������:22
StatefulPartitionedCallStatefulPartitionedCall:v r
N
_output_shapes<
::8������������������������������������
 
_user_specified_nameinputs
�	
�
B__inference_conv_111_layer_call_and_return_conditional_losses_5678

inputs"
conv3d_readvariableop_resource#
biasadd_readvariableop_resource
identity��
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype02
Conv3D/ReadVariableOp�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8������������������������������������*
paddingSAME*
strides	
2
Conv3D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8������������������������������������2	
BiasAdd�
IdentityIdentityBiasAdd:output:0*
T0*N
_output_shapes<
::8������������������������������������2

Identity"
identityIdentity:output:0*U
_input_shapesD
B:8������������������������������������:::v r
N
_output_shapes<
::8������������������������������������
 
_user_specified_nameinputs
�	
�
B__inference_conv3d_5_layer_call_and_return_conditional_losses_4539

inputs"
conv3d_readvariableop_resource#
biasadd_readvariableop_resource
identity��
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype02
Conv3D/ReadVariableOp�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8������������������������������������*
paddingSAME*
strides	
2
Conv3D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8������������������������������������2	
BiasAdd�
IdentityIdentityBiasAdd:output:0*
T0*N
_output_shapes<
::8������������������������������������2

Identity"
identityIdentity:output:0*U
_input_shapesD
B:8������������������������������������:::v r
N
_output_shapes<
::8������������������������������������
 
_user_specified_nameinputs
�	
�
B__inference_conv_111_layer_call_and_return_conditional_losses_4597

inputs"
conv3d_readvariableop_resource#
biasadd_readvariableop_resource
identity��
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype02
Conv3D/ReadVariableOp�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8������������������������������������*
paddingSAME*
strides	
2
Conv3D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8������������������������������������2	
BiasAdd�
IdentityIdentityBiasAdd:output:0*
T0*N
_output_shapes<
::8������������������������������������2

Identity"
identityIdentity:output:0*U
_input_shapesD
B:8������������������������������������:::v r
N
_output_shapes<
::8������������������������������������
 
_user_specified_nameinputs
�
n
B__inference_concat_0_layer_call_and_return_conditional_losses_5529
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis�
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*N
_output_shapes<
::8������������������������������������2
concat�
IdentityIdentityconcat:output:0*
T0*N
_output_shapes<
::8������������������������������������2

Identity"
identityIdentity:output:0*�
_input_shapesv
t:8������������������������������������:8������������������������������������:x t
N
_output_shapes<
::8������������������������������������
"
_user_specified_name
inputs/0:xt
N
_output_shapes<
::8������������������������������������
"
_user_specified_name
inputs/1"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
b
input_0W
serving_default_input_0:08������������������������������������
b
input_1W
serving_default_input_1:08������������������������������������`
add_0W
StatefulPartitionedCall:08������������������������������������tensorflow/serving/predict:��
��
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer_with_weights-4
layer-7
	layer_with_weights-5
	layer-8

layer_with_weights-6

layer-9
layer_with_weights-7
layer-10
layer_with_weights-8
layer-11
layer_with_weights-9
layer-12
layer_with_weights-10
layer-13
layer_with_weights-11
layer-14
layer_with_weights-12
layer-15
layer_with_weights-13
layer-16
layer_with_weights-14
layer-17
layer_with_weights-15
layer-18
layer_with_weights-16
layer-19
layer_with_weights-17
layer-20
layer_with_weights-18
layer-21
layer-22
trainable_variables
regularization_losses
	variables
	keras_api

signatures
+�&call_and_return_all_conditional_losses
�_default_save_signature
�__call__"��
_tf_keras_network��{"class_name": "Functional", "name": "functional_15", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "functional_15", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null, null, null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_0"}, "name": "input_0", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null, null, null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Conv3D", "config": {"name": "conv3d_ind_0_0", "trainable": true, "dtype": "float32", "filters": 15, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv3d_ind_0_0", "inbound_nodes": [[["input_0", 0, 0, {}]]]}, {"class_name": "Conv3D", "config": {"name": "conv3d_ind_0_1", "trainable": true, "dtype": "float32", "filters": 15, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv3d_ind_0_1", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "PReLU", "config": {"name": "prelu_ind_0_0", "trainable": true, "dtype": "float32", "alpha_initializer": {"class_name": "Zeros", "config": {}}, "alpha_regularizer": null, "alpha_constraint": null, "shared_axes": [1, 2, 3]}, "name": "prelu_ind_0_0", "inbound_nodes": [[["conv3d_ind_0_0", 0, 0, {}]]]}, {"class_name": "PReLU", "config": {"name": "prelu_ind_0_1", "trainable": true, "dtype": "float32", "alpha_initializer": {"class_name": "Zeros", "config": {}}, "alpha_regularizer": null, "alpha_constraint": null, "shared_axes": [1, 2, 3]}, "name": "prelu_ind_0_1", "inbound_nodes": [[["conv3d_ind_0_1", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concat_0", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concat_0", "inbound_nodes": [[["prelu_ind_0_0", 0, 0, {}], ["prelu_ind_0_1", 0, 0, {}]]]}, {"class_name": "Conv3D", "config": {"name": "conv3d_0", "trainable": true, "dtype": "float32", "filters": 30, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv3d_0", "inbound_nodes": [[["concat_0", 0, 0, {}]]]}, {"class_name": "PReLU", "config": {"name": "prelu_0", "trainable": true, "dtype": "float32", "alpha_initializer": {"class_name": "Zeros", "config": {}}, "alpha_regularizer": null, "alpha_constraint": null, "shared_axes": [1, 2, 3]}, "name": "prelu_0", "inbound_nodes": [[["conv3d_0", 0, 0, {}]]]}, {"class_name": "Conv3D", "config": {"name": "conv3d_1", "trainable": true, "dtype": "float32", "filters": 30, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv3d_1", "inbound_nodes": [[["prelu_0", 0, 0, {}]]]}, {"class_name": "PReLU", "config": {"name": "prelu_1", "trainable": true, "dtype": "float32", "alpha_initializer": {"class_name": "Zeros", "config": {}}, "alpha_regularizer": null, "alpha_constraint": null, "shared_axes": [1, 2, 3]}, "name": "prelu_1", "inbound_nodes": [[["conv3d_1", 0, 0, {}]]]}, {"class_name": "Conv3D", "config": {"name": "conv3d_2", "trainable": true, "dtype": "float32", "filters": 30, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv3d_2", "inbound_nodes": [[["prelu_1", 0, 0, {}]]]}, {"class_name": "PReLU", "config": {"name": "prelu_2", "trainable": true, "dtype": "float32", "alpha_initializer": {"class_name": "Zeros", "config": {}}, "alpha_regularizer": null, "alpha_constraint": null, "shared_axes": [1, 2, 3]}, "name": "prelu_2", "inbound_nodes": [[["conv3d_2", 0, 0, {}]]]}, {"class_name": "Conv3D", "config": {"name": "conv3d_3", "trainable": true, "dtype": "float32", "filters": 30, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv3d_3", "inbound_nodes": [[["prelu_2", 0, 0, {}]]]}, {"class_name": "PReLU", "config": {"name": "prelu_3", "trainable": true, "dtype": "float32", "alpha_initializer": {"class_name": "Zeros", "config": {}}, "alpha_regularizer": null, "alpha_constraint": null, "shared_axes": [1, 2, 3]}, "name": "prelu_3", "inbound_nodes": [[["conv3d_3", 0, 0, {}]]]}, {"class_name": "Conv3D", "config": {"name": "conv3d_4", "trainable": true, "dtype": "float32", "filters": 30, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv3d_4", "inbound_nodes": [[["prelu_3", 0, 0, {}]]]}, {"class_name": "PReLU", "config": {"name": "prelu_4", "trainable": true, "dtype": "float32", "alpha_initializer": {"class_name": "Zeros", "config": {}}, "alpha_regularizer": null, "alpha_constraint": null, "shared_axes": [1, 2, 3]}, "name": "prelu_4", "inbound_nodes": [[["conv3d_4", 0, 0, {}]]]}, {"class_name": "Conv3D", "config": {"name": "conv3d_5", "trainable": true, "dtype": "float32", "filters": 30, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv3d_5", "inbound_nodes": [[["prelu_4", 0, 0, {}]]]}, {"class_name": "PReLU", "config": {"name": "prelu_5", "trainable": true, "dtype": "float32", "alpha_initializer": {"class_name": "Zeros", "config": {}}, "alpha_regularizer": null, "alpha_constraint": null, "shared_axes": [1, 2, 3]}, "name": "prelu_5", "inbound_nodes": [[["conv3d_5", 0, 0, {}]]]}, {"class_name": "Conv3D", "config": {"name": "conv3d_6", "trainable": true, "dtype": "float32", "filters": 30, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv3d_6", "inbound_nodes": [[["prelu_5", 0, 0, {}]]]}, {"class_name": "PReLU", "config": {"name": "prelu_6", "trainable": true, "dtype": "float32", "alpha_initializer": {"class_name": "Zeros", "config": {}}, "alpha_regularizer": null, "alpha_constraint": null, "shared_axes": [1, 2, 3]}, "name": "prelu_6", "inbound_nodes": [[["conv3d_6", 0, 0, {}]]]}, {"class_name": "Conv3D", "config": {"name": "conv_111", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [1, 1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 1.4142135623730951, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_111", "inbound_nodes": [[["prelu_6", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_0", "trainable": true, "dtype": "float32"}, "name": "add_0", "inbound_nodes": [[["conv_111", 0, 0, {}], ["input_0", 0, 0, {}]]]}], "input_layers": [["input_0", 0, 0], ["input_1", 0, 0]], "output_layers": [["add_0", 0, 0]]}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, null, null, null, 1]}, {"class_name": "TensorShape", "items": [null, null, null, null, 1]}], "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "functional_15", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null, null, null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_0"}, "name": "input_0", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null, null, null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Conv3D", "config": {"name": "conv3d_ind_0_0", "trainable": true, "dtype": "float32", "filters": 15, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv3d_ind_0_0", "inbound_nodes": [[["input_0", 0, 0, {}]]]}, {"class_name": "Conv3D", "config": {"name": "conv3d_ind_0_1", "trainable": true, "dtype": "float32", "filters": 15, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv3d_ind_0_1", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "PReLU", "config": {"name": "prelu_ind_0_0", "trainable": true, "dtype": "float32", "alpha_initializer": {"class_name": "Zeros", "config": {}}, "alpha_regularizer": null, "alpha_constraint": null, "shared_axes": [1, 2, 3]}, "name": "prelu_ind_0_0", "inbound_nodes": [[["conv3d_ind_0_0", 0, 0, {}]]]}, {"class_name": "PReLU", "config": {"name": "prelu_ind_0_1", "trainable": true, "dtype": "float32", "alpha_initializer": {"class_name": "Zeros", "config": {}}, "alpha_regularizer": null, "alpha_constraint": null, "shared_axes": [1, 2, 3]}, "name": "prelu_ind_0_1", "inbound_nodes": [[["conv3d_ind_0_1", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concat_0", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concat_0", "inbound_nodes": [[["prelu_ind_0_0", 0, 0, {}], ["prelu_ind_0_1", 0, 0, {}]]]}, {"class_name": "Conv3D", "config": {"name": "conv3d_0", "trainable": true, "dtype": "float32", "filters": 30, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv3d_0", "inbound_nodes": [[["concat_0", 0, 0, {}]]]}, {"class_name": "PReLU", "config": {"name": "prelu_0", "trainable": true, "dtype": "float32", "alpha_initializer": {"class_name": "Zeros", "config": {}}, "alpha_regularizer": null, "alpha_constraint": null, "shared_axes": [1, 2, 3]}, "name": "prelu_0", "inbound_nodes": [[["conv3d_0", 0, 0, {}]]]}, {"class_name": "Conv3D", "config": {"name": "conv3d_1", "trainable": true, "dtype": "float32", "filters": 30, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv3d_1", "inbound_nodes": [[["prelu_0", 0, 0, {}]]]}, {"class_name": "PReLU", "config": {"name": "prelu_1", "trainable": true, "dtype": "float32", "alpha_initializer": {"class_name": "Zeros", "config": {}}, "alpha_regularizer": null, "alpha_constraint": null, "shared_axes": [1, 2, 3]}, "name": "prelu_1", "inbound_nodes": [[["conv3d_1", 0, 0, {}]]]}, {"class_name": "Conv3D", "config": {"name": "conv3d_2", "trainable": true, "dtype": "float32", "filters": 30, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv3d_2", "inbound_nodes": [[["prelu_1", 0, 0, {}]]]}, {"class_name": "PReLU", "config": {"name": "prelu_2", "trainable": true, "dtype": "float32", "alpha_initializer": {"class_name": "Zeros", "config": {}}, "alpha_regularizer": null, "alpha_constraint": null, "shared_axes": [1, 2, 3]}, "name": "prelu_2", "inbound_nodes": [[["conv3d_2", 0, 0, {}]]]}, {"class_name": "Conv3D", "config": {"name": "conv3d_3", "trainable": true, "dtype": "float32", "filters": 30, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv3d_3", "inbound_nodes": [[["prelu_2", 0, 0, {}]]]}, {"class_name": "PReLU", "config": {"name": "prelu_3", "trainable": true, "dtype": "float32", "alpha_initializer": {"class_name": "Zeros", "config": {}}, "alpha_regularizer": null, "alpha_constraint": null, "shared_axes": [1, 2, 3]}, "name": "prelu_3", "inbound_nodes": [[["conv3d_3", 0, 0, {}]]]}, {"class_name": "Conv3D", "config": {"name": "conv3d_4", "trainable": true, "dtype": "float32", "filters": 30, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv3d_4", "inbound_nodes": [[["prelu_3", 0, 0, {}]]]}, {"class_name": "PReLU", "config": {"name": "prelu_4", "trainable": true, "dtype": "float32", "alpha_initializer": {"class_name": "Zeros", "config": {}}, "alpha_regularizer": null, "alpha_constraint": null, "shared_axes": [1, 2, 3]}, "name": "prelu_4", "inbound_nodes": [[["conv3d_4", 0, 0, {}]]]}, {"class_name": "Conv3D", "config": {"name": "conv3d_5", "trainable": true, "dtype": "float32", "filters": 30, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv3d_5", "inbound_nodes": [[["prelu_4", 0, 0, {}]]]}, {"class_name": "PReLU", "config": {"name": "prelu_5", "trainable": true, "dtype": "float32", "alpha_initializer": {"class_name": "Zeros", "config": {}}, "alpha_regularizer": null, "alpha_constraint": null, "shared_axes": [1, 2, 3]}, "name": "prelu_5", "inbound_nodes": [[["conv3d_5", 0, 0, {}]]]}, {"class_name": "Conv3D", "config": {"name": "conv3d_6", "trainable": true, "dtype": "float32", "filters": 30, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv3d_6", "inbound_nodes": [[["prelu_5", 0, 0, {}]]]}, {"class_name": "PReLU", "config": {"name": "prelu_6", "trainable": true, "dtype": "float32", "alpha_initializer": {"class_name": "Zeros", "config": {}}, "alpha_regularizer": null, "alpha_constraint": null, "shared_axes": [1, 2, 3]}, "name": "prelu_6", "inbound_nodes": [[["conv3d_6", 0, 0, {}]]]}, {"class_name": "Conv3D", "config": {"name": "conv_111", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [1, 1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 1.4142135623730951, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_111", "inbound_nodes": [[["prelu_6", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_0", "trainable": true, "dtype": "float32"}, "name": "add_0", "inbound_nodes": [[["conv_111", 0, 0, {}], ["input_0", 0, 0, {}]]]}], "input_layers": [["input_0", 0, 0], ["input_1", 0, 0]], "output_layers": [["add_0", 0, 0]]}}}
�"�
_tf_keras_input_layer�{"class_name": "InputLayer", "name": "input_0", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null, null, null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null, null, null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_0"}}
�"�
_tf_keras_input_layer�{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null, null, null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null, null, null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
�


kernel
bias
trainable_variables
 regularization_losses
!	variables
"	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Conv3D", "name": "conv3d_ind_0_0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv3d_ind_0_0", "trainable": true, "dtype": "float32", "filters": 15, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 5, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, null, 1]}}
�


#kernel
$bias
%trainable_variables
&regularization_losses
'	variables
(	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Conv3D", "name": "conv3d_ind_0_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv3d_ind_0_1", "trainable": true, "dtype": "float32", "filters": 15, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 5, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, null, 1]}}
�
)shared_axes
	*alpha
+trainable_variables
,regularization_losses
-	variables
.	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "PReLU", "name": "prelu_ind_0_0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "prelu_ind_0_0", "trainable": true, "dtype": "float32", "alpha_initializer": {"class_name": "Zeros", "config": {}}, "alpha_regularizer": null, "alpha_constraint": null, "shared_axes": [1, 2, 3]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 5, "max_ndim": null, "min_ndim": null, "axes": {"4": 15}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, null, 15]}}
�
/shared_axes
	0alpha
1trainable_variables
2regularization_losses
3	variables
4	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "PReLU", "name": "prelu_ind_0_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "prelu_ind_0_1", "trainable": true, "dtype": "float32", "alpha_initializer": {"class_name": "Zeros", "config": {}}, "alpha_regularizer": null, "alpha_constraint": null, "shared_axes": [1, 2, 3]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 5, "max_ndim": null, "min_ndim": null, "axes": {"4": 15}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, null, 15]}}
�
5trainable_variables
6regularization_losses
7	variables
8	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Concatenate", "name": "concat_0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "concat_0", "trainable": true, "dtype": "float32", "axis": -1}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, null, null, null, 15]}, {"class_name": "TensorShape", "items": [null, null, null, null, 15]}]}
�


9kernel
:bias
;trainable_variables
<regularization_losses
=	variables
>	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Conv3D", "name": "conv3d_0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv3d_0", "trainable": true, "dtype": "float32", "filters": 30, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 5, "axes": {"-1": 30}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, null, 30]}}
�
?shared_axes
	@alpha
Atrainable_variables
Bregularization_losses
C	variables
D	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "PReLU", "name": "prelu_0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "prelu_0", "trainable": true, "dtype": "float32", "alpha_initializer": {"class_name": "Zeros", "config": {}}, "alpha_regularizer": null, "alpha_constraint": null, "shared_axes": [1, 2, 3]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 5, "max_ndim": null, "min_ndim": null, "axes": {"4": 30}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, null, 30]}}
�


Ekernel
Fbias
Gtrainable_variables
Hregularization_losses
I	variables
J	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Conv3D", "name": "conv3d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv3d_1", "trainable": true, "dtype": "float32", "filters": 30, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 5, "axes": {"-1": 30}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, null, 30]}}
�
Kshared_axes
	Lalpha
Mtrainable_variables
Nregularization_losses
O	variables
P	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "PReLU", "name": "prelu_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "prelu_1", "trainable": true, "dtype": "float32", "alpha_initializer": {"class_name": "Zeros", "config": {}}, "alpha_regularizer": null, "alpha_constraint": null, "shared_axes": [1, 2, 3]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 5, "max_ndim": null, "min_ndim": null, "axes": {"4": 30}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, null, 30]}}
�


Qkernel
Rbias
Strainable_variables
Tregularization_losses
U	variables
V	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Conv3D", "name": "conv3d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv3d_2", "trainable": true, "dtype": "float32", "filters": 30, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 5, "axes": {"-1": 30}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, null, 30]}}
�
Wshared_axes
	Xalpha
Ytrainable_variables
Zregularization_losses
[	variables
\	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "PReLU", "name": "prelu_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "prelu_2", "trainable": true, "dtype": "float32", "alpha_initializer": {"class_name": "Zeros", "config": {}}, "alpha_regularizer": null, "alpha_constraint": null, "shared_axes": [1, 2, 3]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 5, "max_ndim": null, "min_ndim": null, "axes": {"4": 30}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, null, 30]}}
�


]kernel
^bias
_trainable_variables
`regularization_losses
a	variables
b	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Conv3D", "name": "conv3d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv3d_3", "trainable": true, "dtype": "float32", "filters": 30, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 5, "axes": {"-1": 30}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, null, 30]}}
�
cshared_axes
	dalpha
etrainable_variables
fregularization_losses
g	variables
h	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "PReLU", "name": "prelu_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "prelu_3", "trainable": true, "dtype": "float32", "alpha_initializer": {"class_name": "Zeros", "config": {}}, "alpha_regularizer": null, "alpha_constraint": null, "shared_axes": [1, 2, 3]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 5, "max_ndim": null, "min_ndim": null, "axes": {"4": 30}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, null, 30]}}
�


ikernel
jbias
ktrainable_variables
lregularization_losses
m	variables
n	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Conv3D", "name": "conv3d_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv3d_4", "trainable": true, "dtype": "float32", "filters": 30, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 5, "axes": {"-1": 30}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, null, 30]}}
�
oshared_axes
	palpha
qtrainable_variables
rregularization_losses
s	variables
t	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "PReLU", "name": "prelu_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "prelu_4", "trainable": true, "dtype": "float32", "alpha_initializer": {"class_name": "Zeros", "config": {}}, "alpha_regularizer": null, "alpha_constraint": null, "shared_axes": [1, 2, 3]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 5, "max_ndim": null, "min_ndim": null, "axes": {"4": 30}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, null, 30]}}
�


ukernel
vbias
wtrainable_variables
xregularization_losses
y	variables
z	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Conv3D", "name": "conv3d_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv3d_5", "trainable": true, "dtype": "float32", "filters": 30, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 5, "axes": {"-1": 30}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, null, 30]}}
�
{shared_axes
	|alpha
}trainable_variables
~regularization_losses
	variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "PReLU", "name": "prelu_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "prelu_5", "trainable": true, "dtype": "float32", "alpha_initializer": {"class_name": "Zeros", "config": {}}, "alpha_regularizer": null, "alpha_constraint": null, "shared_axes": [1, 2, 3]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 5, "max_ndim": null, "min_ndim": null, "axes": {"4": 30}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, null, 30]}}
�

�kernel
	�bias
�trainable_variables
�regularization_losses
�	variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Conv3D", "name": "conv3d_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv3d_6", "trainable": true, "dtype": "float32", "filters": 30, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 5, "axes": {"-1": 30}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, null, 30]}}
�
�shared_axes

�alpha
�trainable_variables
�regularization_losses
�	variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "PReLU", "name": "prelu_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "prelu_6", "trainable": true, "dtype": "float32", "alpha_initializer": {"class_name": "Zeros", "config": {}}, "alpha_regularizer": null, "alpha_constraint": null, "shared_axes": [1, 2, 3]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 5, "max_ndim": null, "min_ndim": null, "axes": {"4": 30}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, null, 30]}}
�

�kernel
	�bias
�trainable_variables
�regularization_losses
�	variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�	
_tf_keras_layer�{"class_name": "Conv3D", "name": "conv_111", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv_111", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [1, 1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 1.4142135623730951, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 5, "axes": {"-1": 30}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, null, 30]}}
�
�trainable_variables
�regularization_losses
�	variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Add", "name": "add_0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "add_0", "trainable": true, "dtype": "float32"}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, null, null, null, 1]}, {"class_name": "TensorShape", "items": [null, null, null, null, 1]}]}
�
0
1
#2
$3
*4
05
96
:7
@8
E9
F10
L11
Q12
R13
X14
]15
^16
d17
i18
j19
p20
u21
v22
|23
�24
�25
�26
�27
�28"
trackable_list_wrapper
 "
trackable_list_wrapper
�
0
1
#2
$3
*4
05
96
:7
@8
E9
F10
L11
Q12
R13
X14
]15
^16
d17
i18
j19
p20
u21
v22
|23
�24
�25
�26
�27
�28"
trackable_list_wrapper
�
�layer_metrics
�metrics
trainable_variables
�non_trainable_variables
regularization_losses
 �layer_regularization_losses
�layers
	variables
�__call__
�_default_save_signature
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
-
�serving_default"
signature_map
3:12conv3d_ind_0_0/kernel
!:2conv3d_ind_0_0/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
�layer_metrics
�metrics
trainable_variables
�non_trainable_variables
 regularization_losses
 �layer_regularization_losses
�layers
!	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
3:12conv3d_ind_0_1/kernel
!:2conv3d_ind_0_1/bias
.
#0
$1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
�
�layer_metrics
�metrics
%trainable_variables
�non_trainable_variables
&regularization_losses
 �layer_regularization_losses
�layers
'	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
-:+2prelu_ind_0_0/alpha
'
*0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
*0"
trackable_list_wrapper
�
�layer_metrics
�metrics
+trainable_variables
�non_trainable_variables
,regularization_losses
 �layer_regularization_losses
�layers
-	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
-:+2prelu_ind_0_1/alpha
'
00"
trackable_list_wrapper
 "
trackable_list_wrapper
'
00"
trackable_list_wrapper
�
�layer_metrics
�metrics
1trainable_variables
�non_trainable_variables
2regularization_losses
 �layer_regularization_losses
�layers
3	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�layer_metrics
�metrics
5trainable_variables
�non_trainable_variables
6regularization_losses
 �layer_regularization_losses
�layers
7	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
-:+2conv3d_0/kernel
:2conv3d_0/bias
.
90
:1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
90
:1"
trackable_list_wrapper
�
�layer_metrics
�metrics
;trainable_variables
�non_trainable_variables
<regularization_losses
 �layer_regularization_losses
�layers
=	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
':%2prelu_0/alpha
'
@0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
@0"
trackable_list_wrapper
�
�layer_metrics
�metrics
Atrainable_variables
�non_trainable_variables
Bregularization_losses
 �layer_regularization_losses
�layers
C	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
-:+2conv3d_1/kernel
:2conv3d_1/bias
.
E0
F1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
E0
F1"
trackable_list_wrapper
�
�layer_metrics
�metrics
Gtrainable_variables
�non_trainable_variables
Hregularization_losses
 �layer_regularization_losses
�layers
I	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
':%2prelu_1/alpha
'
L0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
L0"
trackable_list_wrapper
�
�layer_metrics
�metrics
Mtrainable_variables
�non_trainable_variables
Nregularization_losses
 �layer_regularization_losses
�layers
O	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
-:+2conv3d_2/kernel
:2conv3d_2/bias
.
Q0
R1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
Q0
R1"
trackable_list_wrapper
�
�layer_metrics
�metrics
Strainable_variables
�non_trainable_variables
Tregularization_losses
 �layer_regularization_losses
�layers
U	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
':%2prelu_2/alpha
'
X0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
X0"
trackable_list_wrapper
�
�layer_metrics
�metrics
Ytrainable_variables
�non_trainable_variables
Zregularization_losses
 �layer_regularization_losses
�layers
[	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
-:+2conv3d_3/kernel
:2conv3d_3/bias
.
]0
^1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
]0
^1"
trackable_list_wrapper
�
�layer_metrics
�metrics
_trainable_variables
�non_trainable_variables
`regularization_losses
 �layer_regularization_losses
�layers
a	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
':%2prelu_3/alpha
'
d0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
d0"
trackable_list_wrapper
�
�layer_metrics
�metrics
etrainable_variables
�non_trainable_variables
fregularization_losses
 �layer_regularization_losses
�layers
g	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
-:+2conv3d_4/kernel
:2conv3d_4/bias
.
i0
j1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
i0
j1"
trackable_list_wrapper
�
�layer_metrics
�metrics
ktrainable_variables
�non_trainable_variables
lregularization_losses
 �layer_regularization_losses
�layers
m	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
':%2prelu_4/alpha
'
p0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
p0"
trackable_list_wrapper
�
�layer_metrics
�metrics
qtrainable_variables
�non_trainable_variables
rregularization_losses
 �layer_regularization_losses
�layers
s	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
-:+2conv3d_5/kernel
:2conv3d_5/bias
.
u0
v1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
u0
v1"
trackable_list_wrapper
�
�layer_metrics
�metrics
wtrainable_variables
�non_trainable_variables
xregularization_losses
 �layer_regularization_losses
�layers
y	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
':%2prelu_5/alpha
'
|0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
|0"
trackable_list_wrapper
�
�layer_metrics
�metrics
}trainable_variables
�non_trainable_variables
~regularization_losses
 �layer_regularization_losses
�layers
	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
-:+2conv3d_6/kernel
:2conv3d_6/bias
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
�
�layer_metrics
�metrics
�trainable_variables
�non_trainable_variables
�regularization_losses
 �layer_regularization_losses
�layers
�	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
':%2prelu_6/alpha
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
�
�layer_metrics
�metrics
�trainable_variables
�non_trainable_variables
�regularization_losses
 �layer_regularization_losses
�layers
�	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
-:+2conv_111/kernel
:2conv_111/bias
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
�
�layer_metrics
�metrics
�trainable_variables
�non_trainable_variables
�regularization_losses
 �layer_regularization_losses
�layers
�	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�layer_metrics
�metrics
�trainable_variables
�non_trainable_variables
�regularization_losses
 �layer_regularization_losses
�layers
�	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�2�
G__inference_functional_15_layer_call_and_return_conditional_losses_4713
G__inference_functional_15_layer_call_and_return_conditional_losses_5356
G__inference_functional_15_layer_call_and_return_conditional_losses_5216
G__inference_functional_15_layer_call_and_return_conditional_losses_4629�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
__inference__wrapped_model_4116�
���
FullArgSpec
args� 
varargsjargs
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *���
���
H�E
input_08������������������������������������
H�E
input_18������������������������������������
�2�
,__inference_functional_15_layer_call_fn_5010
,__inference_functional_15_layer_call_fn_4862
,__inference_functional_15_layer_call_fn_5484
,__inference_functional_15_layer_call_fn_5420�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
H__inference_conv3d_ind_0_0_layer_call_and_return_conditional_losses_5494�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
-__inference_conv3d_ind_0_0_layer_call_fn_5503�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
H__inference_conv3d_ind_0_1_layer_call_and_return_conditional_losses_5513�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
-__inference_conv3d_ind_0_1_layer_call_fn_5522�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
G__inference_prelu_ind_0_0_layer_call_and_return_conditional_losses_4129�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *D�A
?�<8������������������������������������
�2�
,__inference_prelu_ind_0_0_layer_call_fn_4137�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *D�A
?�<8������������������������������������
�2�
G__inference_prelu_ind_0_1_layer_call_and_return_conditional_losses_4150�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *D�A
?�<8������������������������������������
�2�
,__inference_prelu_ind_0_1_layer_call_fn_4158�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *D�A
?�<8������������������������������������
�2�
B__inference_concat_0_layer_call_and_return_conditional_losses_5529�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
'__inference_concat_0_layer_call_fn_5535�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
B__inference_conv3d_0_layer_call_and_return_conditional_losses_5545�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
'__inference_conv3d_0_layer_call_fn_5554�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
A__inference_prelu_0_layer_call_and_return_conditional_losses_4171�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *D�A
?�<8������������������������������������
�2�
&__inference_prelu_0_layer_call_fn_4179�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *D�A
?�<8������������������������������������
�2�
B__inference_conv3d_1_layer_call_and_return_conditional_losses_5564�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
'__inference_conv3d_1_layer_call_fn_5573�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
A__inference_prelu_1_layer_call_and_return_conditional_losses_4192�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *D�A
?�<8������������������������������������
�2�
&__inference_prelu_1_layer_call_fn_4200�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *D�A
?�<8������������������������������������
�2�
B__inference_conv3d_2_layer_call_and_return_conditional_losses_5583�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
'__inference_conv3d_2_layer_call_fn_5592�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
A__inference_prelu_2_layer_call_and_return_conditional_losses_4213�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *D�A
?�<8������������������������������������
�2�
&__inference_prelu_2_layer_call_fn_4221�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *D�A
?�<8������������������������������������
�2�
B__inference_conv3d_3_layer_call_and_return_conditional_losses_5602�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
'__inference_conv3d_3_layer_call_fn_5611�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
A__inference_prelu_3_layer_call_and_return_conditional_losses_4234�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *D�A
?�<8������������������������������������
�2�
&__inference_prelu_3_layer_call_fn_4242�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *D�A
?�<8������������������������������������
�2�
B__inference_conv3d_4_layer_call_and_return_conditional_losses_5621�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
'__inference_conv3d_4_layer_call_fn_5630�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
A__inference_prelu_4_layer_call_and_return_conditional_losses_4255�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *D�A
?�<8������������������������������������
�2�
&__inference_prelu_4_layer_call_fn_4263�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *D�A
?�<8������������������������������������
�2�
B__inference_conv3d_5_layer_call_and_return_conditional_losses_5640�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
'__inference_conv3d_5_layer_call_fn_5649�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
A__inference_prelu_5_layer_call_and_return_conditional_losses_4276�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *D�A
?�<8������������������������������������
�2�
&__inference_prelu_5_layer_call_fn_4284�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *D�A
?�<8������������������������������������
�2�
B__inference_conv3d_6_layer_call_and_return_conditional_losses_5659�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
'__inference_conv3d_6_layer_call_fn_5668�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
A__inference_prelu_6_layer_call_and_return_conditional_losses_4297�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *D�A
?�<8������������������������������������
�2�
&__inference_prelu_6_layer_call_fn_4305�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *D�A
?�<8������������������������������������
�2�
B__inference_conv_111_layer_call_and_return_conditional_losses_5678�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
'__inference_conv_111_layer_call_fn_5687�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
?__inference_add_0_layer_call_and_return_conditional_losses_5693�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
$__inference_add_0_layer_call_fn_5699�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
8B6
"__inference_signature_wrapper_5076input_0input_1�
__inference__wrapped_model_4116�"#$*09:@EFLQRX]^dijpuv|��������
���
���
H�E
input_08������������������������������������
H�E
input_18������������������������������������
� "T�Q
O
add_0F�C
add_08�������������������������������������
?__inference_add_0_layer_call_and_return_conditional_losses_5693����
���
���
I�F
inputs/08������������������������������������
I�F
inputs/18������������������������������������
� "L�I
B�?
08������������������������������������
� �
$__inference_add_0_layer_call_fn_5699����
���
���
I�F
inputs/08������������������������������������
I�F
inputs/18������������������������������������
� "?�<8�������������������������������������
B__inference_concat_0_layer_call_and_return_conditional_losses_5529����
���
���
I�F
inputs/08������������������������������������
I�F
inputs/18������������������������������������
� "L�I
B�?
08������������������������������������
� �
'__inference_concat_0_layer_call_fn_5535����
���
���
I�F
inputs/08������������������������������������
I�F
inputs/18������������������������������������
� "?�<8�������������������������������������
B__inference_conv3d_0_layer_call_and_return_conditional_losses_5545�9:V�S
L�I
G�D
inputs8������������������������������������
� "L�I
B�?
08������������������������������������
� �
'__inference_conv3d_0_layer_call_fn_5554�9:V�S
L�I
G�D
inputs8������������������������������������
� "?�<8�������������������������������������
B__inference_conv3d_1_layer_call_and_return_conditional_losses_5564�EFV�S
L�I
G�D
inputs8������������������������������������
� "L�I
B�?
08������������������������������������
� �
'__inference_conv3d_1_layer_call_fn_5573�EFV�S
L�I
G�D
inputs8������������������������������������
� "?�<8�������������������������������������
B__inference_conv3d_2_layer_call_and_return_conditional_losses_5583�QRV�S
L�I
G�D
inputs8������������������������������������
� "L�I
B�?
08������������������������������������
� �
'__inference_conv3d_2_layer_call_fn_5592�QRV�S
L�I
G�D
inputs8������������������������������������
� "?�<8�������������������������������������
B__inference_conv3d_3_layer_call_and_return_conditional_losses_5602�]^V�S
L�I
G�D
inputs8������������������������������������
� "L�I
B�?
08������������������������������������
� �
'__inference_conv3d_3_layer_call_fn_5611�]^V�S
L�I
G�D
inputs8������������������������������������
� "?�<8�������������������������������������
B__inference_conv3d_4_layer_call_and_return_conditional_losses_5621�ijV�S
L�I
G�D
inputs8������������������������������������
� "L�I
B�?
08������������������������������������
� �
'__inference_conv3d_4_layer_call_fn_5630�ijV�S
L�I
G�D
inputs8������������������������������������
� "?�<8�������������������������������������
B__inference_conv3d_5_layer_call_and_return_conditional_losses_5640�uvV�S
L�I
G�D
inputs8������������������������������������
� "L�I
B�?
08������������������������������������
� �
'__inference_conv3d_5_layer_call_fn_5649�uvV�S
L�I
G�D
inputs8������������������������������������
� "?�<8�������������������������������������
B__inference_conv3d_6_layer_call_and_return_conditional_losses_5659���V�S
L�I
G�D
inputs8������������������������������������
� "L�I
B�?
08������������������������������������
� �
'__inference_conv3d_6_layer_call_fn_5668���V�S
L�I
G�D
inputs8������������������������������������
� "?�<8�������������������������������������
H__inference_conv3d_ind_0_0_layer_call_and_return_conditional_losses_5494�V�S
L�I
G�D
inputs8������������������������������������
� "L�I
B�?
08������������������������������������
� �
-__inference_conv3d_ind_0_0_layer_call_fn_5503�V�S
L�I
G�D
inputs8������������������������������������
� "?�<8�������������������������������������
H__inference_conv3d_ind_0_1_layer_call_and_return_conditional_losses_5513�#$V�S
L�I
G�D
inputs8������������������������������������
� "L�I
B�?
08������������������������������������
� �
-__inference_conv3d_ind_0_1_layer_call_fn_5522�#$V�S
L�I
G�D
inputs8������������������������������������
� "?�<8�������������������������������������
B__inference_conv_111_layer_call_and_return_conditional_losses_5678���V�S
L�I
G�D
inputs8������������������������������������
� "L�I
B�?
08������������������������������������
� �
'__inference_conv_111_layer_call_fn_5687���V�S
L�I
G�D
inputs8������������������������������������
� "?�<8�������������������������������������
G__inference_functional_15_layer_call_and_return_conditional_losses_4629�"#$*09:@EFLQRX]^dijpuv|��������
���
���
H�E
input_08������������������������������������
H�E
input_18������������������������������������
p

 
� "L�I
B�?
08������������������������������������
� �
G__inference_functional_15_layer_call_and_return_conditional_losses_4713�"#$*09:@EFLQRX]^dijpuv|��������
���
���
H�E
input_08������������������������������������
H�E
input_18������������������������������������
p 

 
� "L�I
B�?
08������������������������������������
� �
G__inference_functional_15_layer_call_and_return_conditional_losses_5216�"#$*09:@EFLQRX]^dijpuv|��������
���
���
I�F
inputs/08������������������������������������
I�F
inputs/18������������������������������������
p

 
� "L�I
B�?
08������������������������������������
� �
G__inference_functional_15_layer_call_and_return_conditional_losses_5356�"#$*09:@EFLQRX]^dijpuv|��������
���
���
I�F
inputs/08������������������������������������
I�F
inputs/18������������������������������������
p 

 
� "L�I
B�?
08������������������������������������
� �
,__inference_functional_15_layer_call_fn_4862�"#$*09:@EFLQRX]^dijpuv|��������
���
���
H�E
input_08������������������������������������
H�E
input_18������������������������������������
p

 
� "?�<8�������������������������������������
,__inference_functional_15_layer_call_fn_5010�"#$*09:@EFLQRX]^dijpuv|��������
���
���
H�E
input_08������������������������������������
H�E
input_18������������������������������������
p 

 
� "?�<8�������������������������������������
,__inference_functional_15_layer_call_fn_5420�"#$*09:@EFLQRX]^dijpuv|��������
���
���
I�F
inputs/08������������������������������������
I�F
inputs/18������������������������������������
p

 
� "?�<8�������������������������������������
,__inference_functional_15_layer_call_fn_5484�"#$*09:@EFLQRX]^dijpuv|��������
���
���
I�F
inputs/08������������������������������������
I�F
inputs/18������������������������������������
p 

 
� "?�<8�������������������������������������
A__inference_prelu_0_layer_call_and_return_conditional_losses_4171�@V�S
L�I
G�D
inputs8������������������������������������
� "L�I
B�?
08������������������������������������
� �
&__inference_prelu_0_layer_call_fn_4179�@V�S
L�I
G�D
inputs8������������������������������������
� "?�<8�������������������������������������
A__inference_prelu_1_layer_call_and_return_conditional_losses_4192�LV�S
L�I
G�D
inputs8������������������������������������
� "L�I
B�?
08������������������������������������
� �
&__inference_prelu_1_layer_call_fn_4200�LV�S
L�I
G�D
inputs8������������������������������������
� "?�<8�������������������������������������
A__inference_prelu_2_layer_call_and_return_conditional_losses_4213�XV�S
L�I
G�D
inputs8������������������������������������
� "L�I
B�?
08������������������������������������
� �
&__inference_prelu_2_layer_call_fn_4221�XV�S
L�I
G�D
inputs8������������������������������������
� "?�<8�������������������������������������
A__inference_prelu_3_layer_call_and_return_conditional_losses_4234�dV�S
L�I
G�D
inputs8������������������������������������
� "L�I
B�?
08������������������������������������
� �
&__inference_prelu_3_layer_call_fn_4242�dV�S
L�I
G�D
inputs8������������������������������������
� "?�<8�������������������������������������
A__inference_prelu_4_layer_call_and_return_conditional_losses_4255�pV�S
L�I
G�D
inputs8������������������������������������
� "L�I
B�?
08������������������������������������
� �
&__inference_prelu_4_layer_call_fn_4263�pV�S
L�I
G�D
inputs8������������������������������������
� "?�<8�������������������������������������
A__inference_prelu_5_layer_call_and_return_conditional_losses_4276�|V�S
L�I
G�D
inputs8������������������������������������
� "L�I
B�?
08������������������������������������
� �
&__inference_prelu_5_layer_call_fn_4284�|V�S
L�I
G�D
inputs8������������������������������������
� "?�<8�������������������������������������
A__inference_prelu_6_layer_call_and_return_conditional_losses_4297��V�S
L�I
G�D
inputs8������������������������������������
� "L�I
B�?
08������������������������������������
� �
&__inference_prelu_6_layer_call_fn_4305��V�S
L�I
G�D
inputs8������������������������������������
� "?�<8�������������������������������������
G__inference_prelu_ind_0_0_layer_call_and_return_conditional_losses_4129�*V�S
L�I
G�D
inputs8������������������������������������
� "L�I
B�?
08������������������������������������
� �
,__inference_prelu_ind_0_0_layer_call_fn_4137�*V�S
L�I
G�D
inputs8������������������������������������
� "?�<8�������������������������������������
G__inference_prelu_ind_0_1_layer_call_and_return_conditional_losses_4150�0V�S
L�I
G�D
inputs8������������������������������������
� "L�I
B�?
08������������������������������������
� �
,__inference_prelu_ind_0_1_layer_call_fn_4158�0V�S
L�I
G�D
inputs8������������������������������������
� "?�<8�������������������������������������
"__inference_signature_wrapper_5076�"#$*09:@EFLQRX]^dijpuv|��������
� 
���
S
input_0H�E
input_08������������������������������������
S
input_1H�E
input_18������������������������������������"T�Q
O
add_0F�C
add_08������������������������������������