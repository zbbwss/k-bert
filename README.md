# k-bert
基于bert4keras 实现k-bert/kbert 来做电商的实体抽取
class VI_Mask(object):
    """定义下三角Attention Mask（语言模型用）
    """
    def compute_attention_bias(self, inputs=None):
        """通过idxs序列的比较来得到对应的mask
        """
        if self.attention_bias is None:
            def vi_mask(s):
                mask = K.cast(s, K.floatx())
                mask=-(1 - mask) * K.infinity()
                mask = K.expand_dims(mask, axis=1)
                mask = K.tile(mask, [1, 12,1, 1])
                return mask
            self.attention_bias = self.apply(
                inputs=self.inputs[-1],
                layer=Lambda,
                function=vi_mask,
                name='Attention-VI-Mask'
            )

        return self.attention_bias

class Kbert(BERT):
    """构建GPT模型
    链接：https://github.com/openai/finetune-transformer-lm
    """
    @insert_arguments(final_activation='softmax')
    @delete_arguments('with_pool', 'with_mlm')
    def __init__(self, **kwargs):
        super(Kbert, self).__init__(**kwargs)
        self.custom_position_ids = True


    def get_inputs(self):
        """BERT的输入是token_ids和segment_ids
        （但允许自行传入位置id，以实现一些特殊需求）
        """
        x_in = self.apply(
            layer=Input, shape=(self.sequence_length,), name='Input-Token'
        )
        inputs = [x_in]

        if self.segment_vocab_size > 0:
            s_in = self.apply(
                layer=Input,
                shape=(self.sequence_length,),
                name='Input-Segment'
            )
            inputs.append(s_in)

        if self.custom_position_ids:
            p_in = self.apply(
                layer=Input,
                shape=(self.sequence_length,),
                name='Input-Position'
            )
            inputs.append(p_in)

        vm_in = self.apply(
            layer=Input,
            shape=(self.sequence_length,self.sequence_length),
            name='Input-Vmask'
        )
        inputs.append(vm_in)

        return inputs

    def apply_embeddings(self, inputs):
        """GPT的embedding是token、position、segment三者embedding之和
        跟BERT的主要区别是三者相加之后没有加LayerNormalization层。
        """
        inputs = inputs[:]
        x = inputs.pop(0)
        s = inputs.pop(0)
        p = inputs.pop(0)
        x = self.apply(
            inputs=x,
            layer=Embedding,
            input_dim=self.vocab_size,
            output_dim=self.embedding_size,
            embeddings_initializer=self.initializer,
            mask_zero=True,
            name='Embedding-Token'
        )
        if 1==1:
            name = 'Embedding-Segment'
            s = self.apply(
                inputs=s,
                layer=Embedding,
                input_dim=2,
                output_dim=self.embedding_size,
                embeddings_initializer=self.initializer,
                name=name
            )
            x = self.apply(
                inputs=[x, s], layer=Add, name='Embedding-Token-Segment'
            )
        x = self.apply(
            inputs=self.simplify([x, p]),
            layer=PositionEmbedding,
            input_dim=self.max_position,
            output_dim=self.embedding_size,
            merge_mode='add',
            hierarchical=self.hierarchical_position,
            embeddings_initializer=self.initializer,
            custom_position_ids=self.custom_position_ids,
            name='Embedding-Position'
        )
        x = self.apply(
            inputs=x,
            layer=Dropout,
            rate=self.dropout_rate,
            name='Embedding-Dropout'
        )
        if self.embedding_size != self.hidden_size:
            x = self.apply(
                inputs=x,
                layer=Dense,
                units=self.hidden_size,
                kernel_initializer=self.initializer,
                name='Embedding-Mapping'
            )

        return x

    def apply_final_layers(self, inputs):
        """剩余部分
        """
        x = inputs

        # Language Model部分
        x = self.apply(
            inputs=x,
            layer=Embedding,
            arguments={'mode': 'dense'},
            name='Embedding-Token'
        )
        x = self.apply(
            inputs=x,
            layer=Activation,
            activation=self.final_activation,
            name='LM-Activation'
        )

        return x

    def load_variable(self, checkpoint, name):
        """加载单个变量的函数
        """
        variable = super(GPT, self).load_variable(checkpoint, name)
        if name == 'gpt/embeddings/word_embeddings':
            return self.load_embeddings(variable)
        else:
            return variable

    def variable_mapping(self):
        """映射到TF版GPT权重格式
        """
        mapping = super(GPT, self).variable_mapping()
        mapping = {
            k: [
                i.replace('bert/', 'gpt/').replace('encoder', 'transformer')
                for i in v
            ]
            for k, v in mapping.items()
        }
        return mapping




#######################
######自定义模型#########
######自定义模型##########
#######################




