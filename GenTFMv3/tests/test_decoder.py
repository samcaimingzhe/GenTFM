import unittest
import torch
from gen_tfm.table import Schema, validate_category_ids
from gen_tfm.decoder import TableDecoder
from gen_tfm.latent import schema_condition

class DecoderTests(unittest.TestCase):
    def test_loss_masks_gradients_and_legal_outputs(self):
        for _ in range(1):
            schema=Schema(2,2,4)
            meta=[dict(n_cont=1,n_cat=1,cat_cardinalities=[3],**schema.metadata_fields())]
            x=torch.zeros(1,3,schema.table_dim)
            x[:,:,0]=torch.tensor([1.,-1.,999.])
            x[0,:2,schema.category_index(0)]=torch.tensor([0,2])
            mask=torch.tensor([[True,True,False]])
            z=torch.randn(1,3,8,requires_grad=True)
            c=schema_condition(meta,schema,'cpu')
            decoder=TableDecoder(8,2,2,4,16)
            for param in decoder.parameters():
                param.data.zero_()
            loss,details=decoder.compute_loss(z,c,x,meta,schema,mask)
            self.assertAlmostEqual(details['continuous_mse'].item(),1.)
            self.assertAlmostEqual(details['categorical_ce'].item(),torch.log(torch.tensor(3.)).item())
            loss.backward()
            self.assertIsNone(z.grad)
            self.assertIsNotNone(decoder.cont_head.bias.grad)
            result=decoder.reconstruct(z,c,meta,schema,mask,sample_categories=True)
            self.assertEqual(result[0,2].abs().sum().item(),0)
            self.assertEqual(result[:,:,1].abs().sum().item(),0)
            self.assertTrue(((result[0,:2,schema.category_index(0)]>=0)&(result[0,:2,schema.category_index(0)]<3)).all())

    def test_raw_contract_rejects_legacy_and_invalid_ids(self):
        with self.assertRaises(TypeError):
            Schema(cat_encoding='binary')
        for values in [torch.tensor([.5]),torch.tensor([-1.]),torch.tensor([3.])]:
            with self.assertRaises(ValueError):
                validate_category_ids(values,3)

    def test_decoder_can_overfit_corresponding_embeddings(self):
        torch.manual_seed(1)
        torch.set_num_threads(1)
        schema=Schema(1,1,2)
        meta=[dict(n_cont=1,n_cat=1,cat_cardinalities=[2],**schema.metadata_fields())]
        z=torch.tensor([[[-1.,0.],[1.,0.]]])
        x=torch.zeros(1,2,schema.table_dim)
        x[:,:,0]=z[:,:,0]
        x[0,:,schema.category_index(0)]=torch.tensor([0,1])
        mask=torch.ones(1,2,dtype=torch.bool)
        c=schema_condition(meta,schema,'cpu')
        model=TableDecoder(2,1,1,2,16)
        opt=torch.optim.Adam(model.parameters(),lr=.03)
        initial,_=model.compute_loss(z,c,x,meta,schema,mask)
        for _ in range(80):
            loss,_=model.compute_loss(z,c,x,meta,schema,mask)
            opt.zero_grad(); loss.backward(); opt.step()
        final,_=model.compute_loss(z,c,x,meta,schema,mask)
        self.assertLess(final.item(),initial.item()*.05)

if __name__=='__main__':
    unittest.main()
